#!/usr/bin/env python3
import sys
import select
import termios
import time
import tty
from typing import Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R_sci
from std_srvs.srv import Trigger
from rfm.action import MoveJoint, MoveSaved, MoveTcp
from rfm.srv import ManagePose

class TeleopKeyboard(Node):
    def __init__(self):
        super().__init__("teleop_keyboard")

        bridge_node_name = str(self.declare_parameter("bridge_node_name", "ur5").value).strip("/")
        self.bridge_prefix = f"/{bridge_node_name}"

        # actions / service
        self.tcp_action = self.declare_parameter("tcp_action", f"{self.bridge_prefix}/move_tcp").value
        self.joint_action = self.declare_parameter("joint_action", f"{self.bridge_prefix}/move_joint").value
        self.saved_action = self.declare_parameter("saved_action", f"{self.bridge_prefix}/move_saved").value
        self.stop_service = self.declare_parameter("stop_service", f"{self.bridge_prefix}/stop").value
        self.manage_pose_service = self.declare_parameter(
            "manage_pose_service", f"{self.bridge_prefix}/manage_pose"
        ).value

        # steps
        self.step_lin = float(self.declare_parameter("step_lin", 0.01).value)     # meters
        self.step_rot = float(self.declare_parameter("step_rot", 0.05).value)     # radians (rotvec magnitude)
        self.step_joint = float(self.declare_parameter("step_joint", 0.1).value)   # radians
        
        # loop rate
        self.loop_hz = float(self.declare_parameter("loop_hz", 20.0).value)
        self.action_timeout = float(self.declare_parameter("action_timeout", 5.0).value)
        self.server_wait_timeout = float(self.declare_parameter("server_wait_timeout", 0.2).value)

        self.act_tcp = ActionClient(self, MoveTcp, self.tcp_action)
        self.act_joint = ActionClient(self, MoveJoint, self.joint_action)
        self.act_saved = ActionClient(self, MoveSaved, self.saved_action)
        self.manage_pose_client = self.create_client(ManagePose, self.manage_pose_service)

        self.stop_client = self.create_client(Trigger, self.stop_service)

        self.get_logger().info("Press 'h' for help, 'SPACE' to stop, 'q' to quit.")

    def _save_current_pose(self, pose_name: str):
        name = (pose_name or "").strip()
        if not name:
            return
        if not self.manage_pose_client.wait_for_service(timeout_sec=self.server_wait_timeout):
            self.get_logger().warn(f"ManagePose service not available: {self.manage_pose_service}")
            return

        req = ManagePose.Request()
        req.cmd = "save"
        req.name = name
        future = self.manage_pose_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.action_timeout)
        if not future.done() or future.result() is None:
            self.get_logger().warn("Failed to save pose (timeout or no response).")
            return

        resp = future.result()
        level = self.get_logger().info if resp.success else self.get_logger().warn
        level(f"Save pose '{name}': {resp.success} ({resp.message})")

    def _publish_delta(self, dx: float, dy: float, dz: float, drv: Tuple[float, float, float]):
        if not self.act_tcp.wait_for_server(timeout_sec=self.server_wait_timeout):
            self.get_logger().warn(f"MoveTcp action server unavailable: {self.tcp_action}")
            return

        qx, qy, qz, qw = R_sci.from_rotvec([float(drv[0]), float(drv[1]), float(drv[2])]).as_quat()

        goal = MoveTcp.Goal()
        goal.relative = True
        goal.target_pose.header.stamp = self.get_clock().now().to_msg()
        goal.target_pose.pose.position.x = float(dx)
        goal.target_pose.pose.position.y = float(dy)
        goal.target_pose.pose.position.z = float(dz)
        goal.target_pose.pose.orientation.x = float(qx)
        goal.target_pose.pose.orientation.y = float(qy)
        goal.target_pose.pose.orientation.z = float(qz)
        goal.target_pose.pose.orientation.w = float(qw)

        self._send_action_goal(self.act_tcp, goal, "MoveTcp")

    def _publish_joint_delta(self, dj: Tuple[float, float, float, float, float, float]):
        if not self.act_joint.wait_for_server(timeout_sec=self.server_wait_timeout):
            self.get_logger().warn(f"MoveJoint action server unavailable: {self.joint_action}")
            return

        goal = MoveJoint.Goal()
        goal.relative = True
        goal.target_joint.position = list(dj)
        self._send_action_goal(self.act_joint, goal, "MoveJoint")

    def _go_saved_pose(self, pose_name: str):
        name = (pose_name or "").strip().lower()
        if not name:
            return
        if not self.act_saved.wait_for_server(timeout_sec=self.server_wait_timeout):
            self.get_logger().warn(f"MoveSaved action server unavailable: {self.saved_action}")
            return
        goal = MoveSaved.Goal()
        goal.pose_name = name
        self._send_action_goal(self.act_saved, goal, "MoveSaved")

    def _send_action_goal(self, client: ActionClient, goal, action_name: str):
        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=self.action_timeout)
        if not send_future.done() or send_future.result() is None:
            self.get_logger().warn(f"{action_name}: goal send timeout/no response.")
            return

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f"{action_name}: goal rejected.")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self.action_timeout)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().warn(f"{action_name}: result timeout/no response.")
            return

        result = result_future.result().result
        if not result.success:
            self.get_logger().warn(f"{action_name} failed: {result.message}")

    def _stop(self):
        if not self.stop_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().warn(f"Stop service not available: {self.stop_service}")
            return
        req = Trigger.Request()
        future = self.stop_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if future.done() and future.result() is not None:
            self.get_logger().info(f"STOP: {future.result().success} ({future.result().message})")

    def _print_help(self):
        print("\n=== UR5 TELEOP KEYS ===")
        print("  w/s : -x / +x")
        print("  a/d : -y / +y")
        print("  r/f : +z / -z")
        print("  z/x : +Rx / -Rx")
        print("  c/v : +Ry / -Ry")
        print("  b/n : +Rz / -Rz")
        print("  1/2 : +base / -base")
        print("  3/4 : +shoulder / -shoulder")
        print("  5/6 : +elbow / -elbow")
        print("  7/8 : +wrist1 / -wrist1")
        print("  9/0 : +wrist2 / -wrist2")
        print("  -/= : +wrist3 / -wrist3")
        print("  ENTER: save current pose")
        print("  l/L  : load saved pose")
        print("  SPACE: stop")
        print("  q    : quit")
        print("=======================\n")


def _get_keys(timeout_sec: float = 0.05, max_read: int = 32):
    """Return list of pressed keys (unique) available within timeout; [] if none."""
    rlist, _, _ = select.select([sys.stdin], [], [], timeout_sec)
    if not rlist:
        return []

    keys = []
    for _ in range(max_read):
        ch = sys.stdin.read(1)
        if not ch:
            break
        keys.append(ch)
        rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not rlist:
            break

    # de-dup while preserving order
    uniq = []
    for k in keys:
        if k not in uniq:
            uniq.append(k)
    return uniq


def main():
    rclpy.init()
    node = TeleopKeyboard()

    # terminal raw mode
    if not sys.stdin.isatty():
        node.get_logger().error("teleop_keyboard must be run in an interactive terminal (TTY).")
        node.destroy_node()
        rclpy.shutdown()
        return

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    period = 1.0 / max(1e-3, float(node.loop_hz))

    try:
        while rclpy.ok():
            t_start = time.monotonic()
                
            rclpy.spin_once(node, timeout_sec=0.0)

            keys = _get_keys()

            # Priority handling (template publishes only one command per tick).
            if "q" in keys:
                break

            # Enter: ask pose name and call ManagePose(cmd="save", name=<pose_name>)
            if "\n" in keys or "\r" in keys:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                try:
                    pose_name = input("Enter pose name: ").strip()
                    print(f"{pose_name} Saved!")
                finally:
                    tty.setcbreak(fd)
                if pose_name:
                    node._save_current_pose(pose_name)
                continue

            # l/L: ask pose name and call MoveSaved(pose_name)
            if "l" in keys or "L" in keys:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                try:
                    pose_name = input("Load pose name: ").strip()
                finally:
                    tty.setcbreak(fd)
                if pose_name:
                    node._go_saved_pose(pose_name)
                continue

            if "h" in keys:
                node._print_help()

            if " " in keys:
                node._stop()
            dx = dy = dz = 0.0
            rvx = rvy = rvz = 0.0
            dj = [0.0] * 6

            # translation
            if "w" in keys:
                dx += -node.step_lin
            if "s" in keys:
                dx += +node.step_lin
            if "a" in keys:
                dy += -node.step_lin
            if "d" in keys:
                dy += +node.step_lin
            if "r" in keys:
                dz += +node.step_lin
            if "f" in keys:
                dz += -node.step_lin

            # rotation (rotvec components, base frame)
            if "z" in keys:
                rvx += +node.step_rot
            if "x" in keys:
                rvx += -node.step_rot
            if "c" in keys:
                rvy += -node.step_rot
            if "v" in keys:
                rvy += +node.step_rot
            if "b" in keys:
                rvz += +node.step_rot
            if "n" in keys:
                rvz += -node.step_rot

       
            if "1" in keys:
                dj[0] += node.step_joint
            if "2" in keys:
                dj[0] -= node.step_joint
            if "3" in keys:
                dj[1] += node.step_joint
            if "4" in keys:
                dj[1] -= node.step_joint
            if "5" in keys:
                dj[2] += node.step_joint
            if "6" in keys:
                dj[2] -= node.step_joint
            if "7" in keys:
                dj[3] += node.step_joint
            if "8" in keys:
                dj[3] -= node.step_joint
            if "9" in keys:
                dj[4] += node.step_joint
            if "0" in keys:
                dj[4] -= node.step_joint
            if "-" in keys:
                dj[5] += node.step_joint
            if "=" in keys:
                dj[5] -= node.step_joint

           
            if any(abs(v) > 0.0 for v in dj):
                node._publish_joint_delta(tuple(dj))

            if (dx != 0.0) or (dy != 0.0) or (dz != 0.0) or (rvx != 0.0) or (rvy != 0.0) or (rvz != 0.0):
                node._publish_delta(dx=dx, dy=dy, dz=dz, drv=(rvx, rvy, rvz))
            dt = time.monotonic() - t_start
            if dt < period:
                time.sleep(period - dt)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()


            
