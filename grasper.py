#!/usr/bin/env python3
from typing import Optional

import numpy as np

from rfm.action import MoveSaved
from robotiq_gripper import RobotiqGripper

STATUS_READY = "READY"
STATUS_PROMPTED = "PROMPTED"
STATUS_GRASPING = "GRASPING"

GRIPPER_HOST = "192.168.0.43"
GRIPPER_PORT = 63352
GRIPPER_TIMEOUT = 0.3
MOVE_SAVED_WAIT_TIMEOUT = 0.05

class Grasper:
    def __init__(self, node):
        self.node = node
        self.bridge_prefix = getattr(self.node, "bridge_prefix", "/ur5")
        self.speed = 255
        self.force = 255
        self.approach_distance = 0.15
        self.lift_distance = 0.1

        self._grasp_last_target_tcp: Optional[np.ndarray] = None

        self.gripper: Optional[RobotiqGripper] = None
        try:
            self.gripper = RobotiqGripper(
                host=GRIPPER_HOST,
                port=GRIPPER_PORT,
                timeout=GRIPPER_TIMEOUT,
            )
            self.gripper.activate()
        except Exception as e:
            self.node.get_logger().warning(f"Robotiq gripper init failed: {e}")

    def reset_for_prompt(self):
        self._grasp_last_target_tcp = None

    @staticmethod
    def _make_tcp_target(R_base: np.ndarray, position: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_base
        T[:3, 3] = position
        return T

    def _set_grasp_sequence_failed(self, reason: str):
        self._grasp_last_target_tcp = None
        self.node._update_warn(True, "grasp_sequence_failed", f"Grasp sequence failed: {reason}")
        self.node._set_status(STATUS_PROMPTED)

    def start_sequence(self, R, p):
        z_axis_base = R[:, 2]
        z_axis_base = z_axis_base / (np.linalg.norm(z_axis_base) + 1e-12)
        p_approach = p + z_axis_base * self.approach_distance

        T_approach = self._make_tcp_target(R, p_approach)
        self._grasp_last_target_tcp = T_approach.copy()
        self.node._set_status(STATUS_GRASPING)

        ok = self.node.send_move_tcp_goal(
            T_approach,
            self.node.get_clock().now().to_msg(),
            result_cb=self._on_grasp_approach_done,
        )
        if not ok:
            self._set_grasp_sequence_failed("approach_goal_send_failed")

    def _on_grasp_approach_done(self, success: bool, message: str):
        if not success:
            self._set_grasp_sequence_failed(f"approach_failed: {message}")
            return

        try:
            close_ok = self.gripper.close(speed=self.speed, force=self.force, wait=True)
            if not close_ok:
                self.node._update_warn(True, "gripper_close_incomplete", "Gripper close did not report stable stop in timeout.")
        except Exception as e:
            self._set_grasp_sequence_failed(f"gripper_close_exception: {e}")
            return

        if self._grasp_last_target_tcp is None:
            self._set_grasp_sequence_failed("tcp_pose_missing_before_lift")
            return

        p_lift = self._grasp_last_target_tcp[:3, 3].copy()
        p_lift[2] += self.lift_distance
        R_lift = self._grasp_last_target_tcp[:3, :3].copy()
        T_lift = self._make_tcp_target(R_lift, p_lift)

        self._grasp_last_target_tcp = T_lift.copy()
        ok = self.node.send_move_tcp_goal(
            T_lift,
            self.node.get_clock().now().to_msg(),
            result_cb=self._on_grasp_lift_done,
        )
        if not ok:
            self._set_grasp_sequence_failed("lift_goal_send_failed")

    def _on_grasp_lift_done(self, success: bool, message: str):
        if self.gripper is None:
            self._set_grasp_sequence_failed("gripper_missing_after_lift")
            return

        if not success:
            self.node._update_warn(True, "grasp_lift_failed", f"Lift motion failed: {message}")

        try:
            open_ok = self.gripper.open(speed=self.speed, force=self.force, wait=True)
            if not open_ok:
                self.node._update_warn(True, "gripper_open_incomplete", "Gripper open did not reach target in timeout.")
        except Exception as e:
            self._set_grasp_sequence_failed(f"gripper_open_exception: {e}")
            return

        self._grasp_last_target_tcp = None
        self.node._update_warn(False, "grasp_sequence_failed")
        self.node._set_status(STATUS_READY)
        self.node.on_prompt_succeeded()

        try:
            if self.node.act_move_saved.wait_for_server(timeout_sec=MOVE_SAVED_WAIT_TIMEOUT):
                goal = MoveSaved.Goal()
                goal.pose_name = "observe"
                self.node.act_move_saved.send_goal_async(goal)
            else:
                self.node._update_warn(
                    True,
                    "move_saved_server_unavailable",
                    f"MoveSaved action server unavailable: {self.bridge_prefix}/move_saved",
                )
        except Exception as e:
            self.node._update_warn(True, "move_saved_send_failed", f"MoveSaved goal send failed: {e}")
