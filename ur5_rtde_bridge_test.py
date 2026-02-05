import threading
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger

from rclpy.time import Time
from math_utils import quat_to_mat, mat_to_quat, mat_to_rotvec, rotvec_to_mat

import json
import os



class UR5RTDEBridge(Node):
    def __init__(self):
        super().__init__("ur5_rtde_bridge")

        self.robot_ip = self.declare_parameter("robot_ip", "192.168.0.43").value
        self.speed = float(self.declare_parameter("speed", 0.1).value)
        self.accel = float(self.declare_parameter("accel", 0.1).value)
        self.tcp_frame = self.declare_parameter("tcp_frame", "base_link").value
        self.publish_rate_hz = float(self.declare_parameter("publish_rate_hz", 30.0).value)

        import rtde_control, rtde_receive
        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)

        self.exec_status = "IDLE"
        self.moving = False
        self.lock = threading.Lock()

        self.pub_tcp_pose = self.create_publisher(PoseStamped, "tcp_pose", 10)
        self.pub_exec_status = self.create_publisher(String, "exec_status", 10)

        # subscribe PoseTracker output
        self.sub_target = self.create_subscription(
            PoseStamped, "/test/goal_tcp_pose", self._on_target, 10
        )

        self.sub_relation_target = self.create_subscription(
            PoseStamped, "/test/goal_tcp_pose", self._on_relation_target, 10
        )

        # stop service
        self.srv_stop = self.create_service(Trigger, "stop", self._on_stop)

        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._publish_state)

        self.get_logger().info(
            f"UR5 RTDE bridge connected to {self.robot_ip}. Subscribing /test/goal_tcp_pose"
        )

        # ---- command topic + pose DB (ADDED) ----
        self.pose_db_path = self.declare_parameter(
            "pose_db_path",
            os.path.expanduser("~/.ur5_saved_poses.json")
        ).value
        self.pose_db = {}
        self._load_pose_db()

        self.sub_cmd = self.create_subscription(
            String, "/ur5_rtde_bridge/cmd", self._on_cmd, 10
        )

    # ---- command handlers (ADDED) ----
    def _on_cmd(self, msg: String):
        line = (msg.data or "").strip()
        if not line:
            return

        parts = line.split()
        cmd = parts[0].lower()
        arg = parts[1].lower() if len(parts) >= 2 else None

        if cmd == "where":
            pose = self.rtde_r.getActualTCPPose()
            self.get_logger().info(f"CURRENT TCP: {pose}")
            return

        if cmd == "list":
            keys = sorted(self.pose_db.keys())
            self.get_logger().info(f"SAVED: {keys}")
            return

        if cmd == "save":
            if not arg:
                self.get_logger().warn("Usage: save <name>")
                return
            pose = self.rtde_r.getActualTCPPose()
            self.pose_db[arg] = [float(v) for v in pose]
            self._save_pose_db()
            self.get_logger().info(f"SAVED '{arg}': {self.pose_db[arg]}")
            return

        if cmd in ("go", "observe", "three"):
            name = arg if cmd == "go" else cmd
            if not name:
                self.get_logger().warn("Usage: go <name>")
                return
            self._go_saved_pose(name)
            return

        self.get_logger().warn(f"Unknown command: {cmd}")

    def _go_saved_pose(self, name: str):
        if name not in self.pose_db:
            self.get_logger().warn(f"No saved pose: '{name}'. Use 'list' or 'save {name}'.")
            return

        with self.lock:
            if self.moving:
                self.get_logger().warn("Robot is moving. Try again after it stops.")
                return
            self.moving = True
            self.exec_status = "MOVING"

        target = list(self.pose_db[name])

        def worker():
            try:
                self.get_logger().info(f"GO '{name}' -> {target}")
                ok = self.rtde_c.moveL(target, self.speed, self.accel)
                self.exec_status = "SUCCESS" if ok else "FAILED"
            except Exception as e:
                self.exec_status = "FAILED"
                self.get_logger().error(str(e))
            finally:
                with self.lock:
                    self.moving = False

        threading.Thread(target=worker, daemon=True).start()

    def _load_pose_db(self):
        try:
            if os.path.exists(self.pose_db_path):
                with open(self.pose_db_path, "r") as f:
                    data = json.load(f)
                for k, v in data.items():
                    if isinstance(v, list) and len(v) == 6:
                        self.pose_db[str(k).lower()] = [float(x) for x in v]
            self.get_logger().info(f"Loaded pose DB: {self.pose_db_path} (keys={list(self.pose_db.keys())})")
        except Exception as e:
            self.get_logger().warn(f"Failed to load pose DB: {e}")

    def _save_pose_db(self):
        try:
            os.makedirs(os.path.dirname(self.pose_db_path), exist_ok=True)
            with open(self.pose_db_path, "w") as f:
                json.dump(self.pose_db, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"Failed to save pose DB: {e}")
    # -----------------------------------
    # -----------------------------------

    def _publish_state(self):
        # publish exec_status
        s = String()
        s.data = self.exec_status
        self.pub_exec_status.publish(s)

        # publish tcp_pose
        pose = self.rtde_r.getActualTCPPose()  # [x,y,z,rx,ry,rz]
        x,y,z,rx,ry,rz = pose
        R = rotvec_to_mat(np.array([rx,ry,rz], dtype=np.float64))
        q = mat_to_quat(R)

        msg = PoseStamped()
        msg.header.frame_id = self.tcp_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        self.pub_tcp_pose.publish(msg)

    def _on_stop(self, req, resp):
        try:
            self.rtde_c.stopL(0.5)
            self.exec_status = "IDLE"
            resp.success = True
            resp.message = "stopL called"
        except Exception as e:
            resp.success = False
            resp.message = str(e)
        return resp

    def _on_target(self, msg: PoseStamped):
        with self.lock:
            if self.moving:
                return
            self.moving = True
            self.exec_status = "MOVING"

        # spawn thread so ROS callback doesn't block
        def worker():
            try:
                p = msg.pose.position
                q = msg.pose.orientation
                R = quat_to_mat([q.x,q.y,q.z,q.w])
                rv = mat_to_rotvec(R)
                target = [float(p.x), float(p.y), float(p.z), float(rv[0]), float(rv[1]), float(rv[2])]
                # 1-line UR motion (moveL)
                ok = self.rtde_c.moveL(target, self.speed, self.accel)
                self.exec_status = "SUCCESS" if ok else "FAILED"
            except Exception:
                self.exec_status = "FAILED"
            finally:
                with self.lock:
                    self.moving = False

        threading.Thread(target=worker, daemon=True).start()

    def _on_relation_target(self, msg: PoseStamped):
        with self.lock:
            if self.moving:
                return
            self.moving = True
            self.exec_status = "MOVING"

        # spawn thread so ROS callback doesn't block
        def worker():
            try:     
                p = msg.pose.position
                q = msg.pose.orientation
                R = quat_to_mat([q.x,q.y,q.z,q.w])
                rv = mat_to_rotvec(R)
                target = [float(p.x), float(p.y), float(p.z), float(rv[0]), float(rv[1]), float(rv[2])]

                pose = self.rtde_r.getActualTCPPose()
                print("current : ", pose)

                
                # x 방향으로 p만큼
                pose[0] += p.x
                pose[1] += p.y
                pose[2] += p.z
                pose[3] += rv[0]
                pose[4] += rv[1]
                pose[5] += rv[2]


                ok = self.rtde_c.moveL(pose, speed=0.2, acceleration=0.4)
                self.exec_status = "SUCCESS" if ok else "FAILED"

            except Exception:
                self.exec_status = "FAILED"
            finally:
                with self.lock:
                    self.moving = False

        threading.Thread(target=worker, daemon=True).start()





def main():
    rclpy.init()
    node = UR5RTDEBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

