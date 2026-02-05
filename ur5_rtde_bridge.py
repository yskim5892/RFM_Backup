import threading
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger

from rclpy.time import Time
from math_utils import quat_to_mat, mat_to_quat, mat_to_rotvec, rotvec_to_mat


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
            PoseStamped, "/pose_tracker/goal_tcp_pose", self._on_target, 10
        )

        self.sub_relation_target = self.create_subscription(
            PoseStamped, "/pose_tracker/goal_tcp_pose_r", self._on_relation_target, 10
        )

        # stop service
        self.srv_stop = self.create_service(Trigger, "stop", self._on_stop)

        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._publish_state)

        self.get_logger().info(
            f"UR5 RTDE bridge connected to {self.robot_ip}. Subscribing /pose_tracker/goal_tcp_pose"
        )

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
                print(pose)
                print(rv[2])
                
                # x 방향으로 p만큼
                pose[0] -= p.x
                pose[1] -= p.y
                pose[2] -= p.z
                pose[3] -= rv[0]
                pose[4] -= rv[1]
                pose[5] -= rv[2]

                ok = self.rtde_c.moveL(pose, speed=self.speed, acceleration=self.accel)
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

