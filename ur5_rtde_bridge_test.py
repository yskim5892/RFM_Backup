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
        self.tcp_frame = self.declare_parameter("tcp_frame", "base").value
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
            PoseStamped, "/test/goal_tcp_pose_r", self._on_relation_target, 10
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

        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip().lower() if len(parts) == 2 else None

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

        if cmd == "go":
            if not arg:
                self.get_logger().warn("Usage: go <name>")
                return
            if arg not in self.pose_db:
                self.get_logger().warn(f"No saved pose: '{arg}'. Use 'list' or 'save {arg}'.")
                return
            self._go_saved_pose(arg)
            return

        self.get_logger().warn("Unknown command. Supported: where, list, save <name>, go <name>")


    def _start_motion(self, worker_fn, busy_msg: str = "Robot is moving. Try again after it stops."):
        with self.lock:
            if self.moving:
                self.get_logger().warn(busy_msg)
                return
            self.moving = True
            self.exec_status = "MOVING"

        def runner():
            try:
                ok = bool(worker_fn())
                self.exec_status = "SUCCESS" if ok else "FAILED"
            except Exception as e:
                self.exec_status = "FAILED"
                self.get_logger().error(str(e))
            finally:
                with self.lock:
                    self.moving = False

        threading.Thread(target=runner, daemon=True).start()

    @staticmethod
    def _pose_to_rtde_target(msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        R = quat_to_mat([q.x, q.y, q.z, q.w])
        rv = mat_to_rotvec(R)
        return [float(p.x), float(p.y), float(p.z), float(rv[0]), float(rv[1]), float(rv[2])]

    def _go_saved_pose(self, name: str):
        if name not in self.pose_db:
            self.get_logger().warn(f"No saved pose: '{name}'. Use 'list' or 'save {name}'.")
            return

        target = list(self.pose_db[name])
        self.get_logger().info(f"GO '{name}' -> {target}")
        self._start_motion(lambda: self.rtde_c.moveL(target, self.speed, self.accel))

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
        target = self._pose_to_rtde_target(msg)
        self._start_motion(lambda: self.rtde_c.moveL(target, self.speed, self.accel),
                           busy_msg="Robot is moving. Ignore absolute target.")

    def _on_relation_target(self, msg: PoseStamped):
        delta = self._pose_to_rtde_target(msg)

        def worker():
            pose = self.rtde_r.getActualTCPPose()
            for i in range(6):
                pose[i] += delta[i]
            return self.rtde_c.moveL(pose, speed=self.speed, acceleration=self.accel)

        self._start_motion(worker, busy_msg="Robot is moving. Ignore relative target.")

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

