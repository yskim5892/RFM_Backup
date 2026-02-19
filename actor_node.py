#!/usr/bin/env python3
from collections import deque
import json
import threading
import traceback
from pathlib import Path
from typing import Optional

import cv2
import message_filters
import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

from grasper import Grasper
from math_utils import quat_to_mat, tf_to_T
from pose_tracker import PoseTracker
from publish_static_tf import StaticTFPublisher
from rfm.action import MoveSaved, MoveTcp

import utils

STATUS_READY = "READY"
STATUS_MISSING_INPUT = "MISSING_INPUT"
STATUS_ERROR = "ERROR"
STATUS_PROMPTED = "PROMPTED"
STATUS_APPROACHING = "APPROACHING"
STATUS_GRASPING = "GRASPING"


class ActorNode(Node):
    def __init__(self):
        super().__init__("actor")

        self.lock = threading.Lock()
        self.busy = False
        self.current_status = STATUS_READY

        repo_root = Path(__file__).resolve().parent
        self.repo_root = repo_root

        self.base_frame = self.declare_parameter("base_frame", "base").value
        self.camera_frame = self.declare_parameter("camera_frame", "camera_color_optical_frame").value
        self.depth_scale = 0.001

        self.min_mask_pixels = int(self.declare_parameter("min_mask_pixels", 200).value)
        self.sync_queue_size = int(self.declare_parameter("sync_queue_size", 30).value)
        self.sync_slop_sec = float(self.declare_parameter("sync_slop_sec", 0.06).value)
        self.tf_lookup_timeout_sec = float(self.declare_parameter("tf_lookup_timeout_sec", 0.1).value)
        self.skip_while_moving = bool(self.declare_parameter("skip_while_moving", False).value)

        self.pregrasp_height = float(self.declare_parameter("pregrasp_height", 0.3).value)

        self.color_topic = "/wrist_cam/camera/color/image_raw"
        self.depth_topic = "/wrist_cam/camera/aligned_depth_to_color/image_raw"
        self.instance_mask_topic = "/perception/target_object_mask"
        self.initial_prompt = self.declare_parameter("prompt", "").value

        self.static_tf_file = str(
            self.declare_parameter("static_tf_file", str(self.repo_root / "T_wrist_cam.txt")).value
        )
        bridge_node_name = str(self.declare_parameter("bridge_node_name", "ur5").value).strip("/")
        self.bridge_prefix = f"/{bridge_node_name}"
        self.ur5_tcp_topic = self.declare_parameter("ur5_tcp_topic", f"{self.bridge_prefix}/tcp_pose").value
        self.ur5_status_topic = self.declare_parameter("ur5_status_topic", f"{self.bridge_prefix}/status").value
        self.ur5_move_tcp_action = self.declare_parameter("ur5_move_tcp_action", f"{self.bridge_prefix}/move_tcp").value
        self.gsam_prompt_dequeue_srv = self.declare_parameter(
            "gsam_prompt_dequeue_srv", "/perception/dequeue_prompt"
        ).value
        self.ur5_action_wait_timeout = float(self.declare_parameter("ur5_action_wait_timeout", 0.05).value)

        self.grasp_depth_threshold_m = float(self.declare_parameter("grasp_depth_threshold_m", 0.4).value)
        self.grasp_center_dist_px_threshold = float(self.declare_parameter("grasp_center_dist_px_threshold", 30.0).value)
        self.grasp_center_y = float(self.declare_parameter("grasp_center_y", 0.2).value)
        self.grasp_center_y_weight = float(self.declare_parameter("grasp_center_y_weight", 0.5).value)

        strategy_default = str(self.repo_root / "grasp_strategy.json")
        self.grasp_strategy_file = Path(self.declare_parameter("grasp_strategy_file", strategy_default).value)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.static_tf_publisher = StaticTFPublisher(self)
        self.static_tf_publisher.publish_from_file(matrix_file=self.static_tf_file, parent="tool0", child="camera_link")

        self.pub_T_base_obj = self.create_publisher(PoseStamped, "/pose_tracker/T_base_obj", 10)
        self.pub_T_cam_obj = self.create_publisher(PoseStamped, "/pose_tracker/T_cam_obj", 10)
        self.pub_target_tcp = self.create_publisher(PoseStamped, "/pose_tracker/target_tcp_pose", 10)
        self.pub_pose_vis = self.create_publisher(Image, "/pose_tracker/pose_vis", 10)

        status_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_status = self.create_publisher(String, "/pose_tracker/status", status_qos)

        self.sub_tcp_pose = self.create_subscription(PoseStamped, self.ur5_tcp_topic, self._on_tcp_pose, 10)
        self.sub_ur5_status = self.create_subscription(String, self.ur5_status_topic, self._on_ur5_status, 10)
        self.sub_prompt = self.create_subscription(String, "/inference/prompt", self._on_prompt, 10)
        self.sub_cam_info = self.create_subscription(CameraInfo, "/wrist_cam/camera/color/camera_info", self._on_cam_info, 10)
        self.sub_target_mask_arrived = self.create_subscription(Image, self.instance_mask_topic, self._on_target_mask_arrived, qos_profile_sensor_data)

        self._latest_tcp_R_base = np.eye(3, dtype=np.float64)
        self._latest_tcp_p_base = np.zeros(3, dtype=np.float64)
        self.ur5_status = "IDLE"

        self.act_move_tcp = ActionClient(self, MoveTcp, self.ur5_move_tcp_action)
        self.act_move_saved = ActionClient(self, MoveSaved, f"{self.bridge_prefix}/move_saved")
        self.cli_gsam_prompt_dequeue = self.create_client(Trigger, self.gsam_prompt_dequeue_srv)
        self._move_tcp_result_cb = None

        self.pose_tracker = PoseTracker(self.repo_root, self.get_logger())
        self.grasp_strategy_keys = self._load_grasp_strategy_keys(self.grasp_strategy_file)

        self.grasper = Grasper(self)

        self._prompt_queue: deque[str] = deque()
        self._active_prompt_obj: Optional[str] = None
        self._target_mask_msg_count = 0
        self._required_target_mask_count: Optional[int] = None
        self._warned_once_keys: set[str] = set()

        self._use_foundation_pose = False
        self._last_grasp_metric_log_ns = 0

        self.K: Optional[np.ndarray] = None
        self.srv_reset = self.create_service(Trigger, "/pose_tracker/reset", self._on_reset)

        self.color_sub = message_filters.Subscriber(self, Image, self.color_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.mask_sub = message_filters.Subscriber(self, Image, self.instance_mask_topic, qos_profile=qos_profile_sensor_data)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.mask_sub],
            self.sync_queue_size,
            self.sync_slop_sec,
        )
        self.sync.registerCallback(self._on_synced)
        self._synced_once = False

        self._publish_status()

        self.get_logger().info(
            "ActorNode ready.\n"
            f"  Strategy file: {self.grasp_strategy_file}\n"
            f"  Strategy keys: {sorted(self.grasp_strategy_keys)}\n"
            f"  Sub color: {self.color_topic}\n"
            f"  Sub depth: {self.depth_topic}\n"
            f"  Sub mask : {self.instance_mask_topic}\n"
            f"  Action MoveTcp: {self.ur5_move_tcp_action}\n"
            "  Pub /pose_tracker/T_base_obj, /pose_tracker/T_cam_obj, /pose_tracker/target_tcp_pose, /pose_tracker/status"
        )

        if self.initial_prompt:
            self._on_prompt(String(data=self.initial_prompt))

    def _load_grasp_strategy_keys(self, json_path: Path) -> set[str]:
        if not json_path.exists():
            self.get_logger().warning(f"grasp strategy file not found: {json_path}")
            return set()

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            self.get_logger().warning(f"failed to read grasp strategy file: {e}")
            return set()

        if not isinstance(data, dict):
            self.get_logger().warning("grasp strategy file should be a JSON object")
            return set()

        return {self.pose_tracker.normalize_object_name(k) for k in data.keys()}

    def _on_prompt(self, msg: String):
        obj_name = utils.parse_prompt_to_object(msg.data)

        self._prompt_queue.append(obj_name)
        self.get_logger().info(f"Prompt queued: '{obj_name}' (queue={len(self._prompt_queue)})")
        self._try_activate_next_prompt()

    def _try_activate_next_prompt(self):
        if self._active_prompt_obj is not None:
            return
        if not self._prompt_queue:
            self._set_status(STATUS_READY)
            return

        obj_name = self._prompt_queue[0]
        norm = self.pose_tracker.normalize_object_name(obj_name)
        has_strategy = norm in self.grasp_strategy_keys
        has_mesh = self.pose_tracker.mesh_exists(obj_name)

        use_fp = has_strategy and has_mesh
        if use_fp:
            try:
                use_fp = self.pose_tracker.set_object(obj_name)
            except Exception as e:
                self.get_logger().warning(f"FoundationPose object setup failed. fallback to mask+depth: {e}")
                use_fp = False

        if not use_fp:
            self.pose_tracker.clear_object()

        self.pose_tracker.reset_tracking()
        self._use_foundation_pose = use_fp
        self._required_target_mask_count = self._target_mask_msg_count + 1
        self.grasper.reset_for_prompt()
        self._move_tcp_result_cb = None
        self._active_prompt_obj = obj_name

        mode = "FoundationPose" if self._use_foundation_pose else "MaskDepth"
        self.get_logger().info(
            f"Prompt activated: object='{norm}', strategy={has_strategy}, mesh={has_mesh}, mode={mode}"
        )
        self._set_status(STATUS_PROMPTED)

    def _notify_gsam_prompt_dequeue_async(self):
        if not self.cli_gsam_prompt_dequeue.wait_for_service(timeout_sec=0.2):
            self.get_logger().warning(f"{self.gsam_prompt_dequeue_srv} service not available")
            return
        req = Trigger.Request()
        fut = self.cli_gsam_prompt_dequeue.call_async(req)
        fut.add_done_callback(self._on_gsam_prompt_dequeue_done)

    def _on_gsam_prompt_dequeue_done(self, fut):
        try:
            res = fut.result()
            if not res.success:
                self.get_logger().warning(f"GSAM dequeue failed: {res.message}")
        except Exception as e:
            self.get_logger().warning(f"GSAM dequeue call exception: {e}")

    def on_prompt_succeeded(self):
        if self._active_prompt_obj is None:
            return

        if self._prompt_queue:
            self._prompt_queue.popleft()

        self._active_prompt_obj = None
        self._notify_gsam_prompt_dequeue_async()
        self._try_activate_next_prompt()

    def _publish_status(self):
        msg = String()
        msg.data = self.current_status
        self.pub_status.publish(msg)

    def _set_status(self, s: str):
        if s != self.current_status:
            self.current_status = s
            self._publish_status()

    def _update_warn(self, flag: bool, key: str, msg: str = "") -> bool:
        if flag:
            if key in self._warned_once_keys:
                return True
            self._warned_once_keys.add(key)
            self.get_logger().warning(msg)
            return True
        self._warned_once_keys.discard(key)
        return False

    def _on_reset(self, req, resp):
        self.pose_tracker.reset_tracking()
        self.grasper.reset_for_prompt()
        self._move_tcp_result_cb = None
        if self._active_prompt_obj is not None:
            self._required_target_mask_count = self._target_mask_msg_count + 1
            self._set_status(STATUS_PROMPTED)
        else:
            self._set_status(STATUS_READY)
        resp.success = True
        resp.message = "reset ok"
        return resp

    def _on_cam_info(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _on_tcp_pose(self, msg: PoseStamped):
        q = msg.pose.orientation
        self._latest_tcp_R_base = quat_to_mat([q.x, q.y, q.z, q.w])
        self._latest_tcp_p_base = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def _on_ur5_status(self, msg: String):
        self.ur5_status = msg.data

    def _on_target_mask_arrived(self, _msg: Image):
        self._target_mask_msg_count += 1

    def _estimate_center_from_mask_depth(self, mask, depth, K):
        mask_u8 = mask.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask_er = cv2.erode(mask_u8, kernel, iterations=1).astype(bool)

        ys, xs = np.nonzero(mask_er)

        depth_roi = depth[mask_er]
        valid_depth = depth_roi[np.isfinite(depth_roi) & (depth_roi > 1e-6)]
        if xs.size == 0 or valid_depth.size == 0:
            return None, float("nan"), float("nan"), None

        u, v = float(xs.mean()), float(ys.mean())
        z = float(np.median(valid_depth))
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        p_obj_cam = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)
        dx = u - 0.5 * float(mask.shape[1] - 1)
        dy = v - self.grasp_center_y * float(mask.shape[0] - 1)
        center_dist_px = float(np.hypot(dx, self.grasp_center_y_weight * dy))

        return p_obj_cam, z, center_dist_px, (int(round(u)), int(round(v)))

    def _compute_target_tcp_from_point(self, p_obj_base: np.ndarray) -> Optional[np.ndarray]:
        R_tcp = self._latest_tcp_R_base
        z_wrist_local = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        z_wrist_base = R_tcp @ z_wrist_local
        z_wrist_base = z_wrist_base / (np.linalg.norm(z_wrist_base) + 1e-12)

        p_tcp = p_obj_base + z_wrist_base * self.pregrasp_height
        if p_tcp[2] < 0.2:
            p_tcp[2] = 0.2

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_tcp
        T[:3, 3] = p_tcp
        return T

    def _run_grasp_control(self, z_mean: float, center_dist_px: float, target_tcp: Optional[np.ndarray], stamp) -> None:
        if self.current_status != STATUS_PROMPTED or self.ur5_status != "IDLE" or target_tcp is None:
            return

        if (np.isfinite(z_mean)
            and (z_mean <= self.grasp_depth_threshold_m)
            and np.isfinite(center_dist_px)
            and (center_dist_px <= self.grasp_center_dist_px_threshold)):
            self.grasper.start_sequence(self._latest_tcp_R_base, self._latest_tcp_p_base)
            return

        self._publish_pose(self.pub_target_tcp, self.base_frame, stamp, target_tcp)
        if self.send_move_tcp_goal(target_tcp, stamp, result_cb=self._on_approach_done):
            self._set_status(STATUS_APPROACHING)

    def _log_grasp_gate(self, z_mean: float, center_dist_px: float):
        now_ns = int(self.get_clock().now().nanoseconds)
        if now_ns - self._last_grasp_metric_log_ns <= int(1.5 * 1e9):
            return
        self._last_grasp_metric_log_ns = now_ns
        self.get_logger().info(
            f"grasp_gate z_mean={z_mean:.3f}m (<= {self.grasp_depth_threshold_m:.3f}), "
            f"center_dist={center_dist_px:.1f}px (<= {self.grasp_center_dist_px_threshold:.1f}), "
            f"status={self.current_status}"
        )

    def _publish_pose(self, topic_pub, frame_id: str, stamp, T: np.ndarray):
        topic_pub.publish(utils.make_pose_msg(frame_id, stamp, T))

    def send_move_tcp_goal(self, T_base_tcp: np.ndarray, stamp, result_cb=None) -> bool:
        if self._update_warn(
            not self.act_move_tcp.wait_for_server(timeout_sec=self.ur5_action_wait_timeout),
            "move_tcp_server_unavailable",
            f"MoveTcp action server unavailable: {self.ur5_move_tcp_action}",
        ):
            return False

        goal = MoveTcp.Goal()
        goal.relative = False
        goal.target_pose = utils.make_pose_msg(self.base_frame, stamp, T_base_tcp)

        self._move_tcp_result_cb = result_cb
        send_future = self.act_move_tcp.send_goal_async(goal)
        send_future.add_done_callback(self._on_move_tcp_goal_response)
        return True

    def _on_move_tcp_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self._update_warn(True, "move_tcp_send_failed", f"MoveTcp goal send failed: {e}")
            self._run_move_tcp_result_cb(False, str(e))
            return
        self._update_warn(False, "move_tcp_send_failed")

        if self._update_warn(not goal_handle.accepted, "move_tcp_goal_rejected", "MoveTcp goal rejected by action server."):
            self._run_move_tcp_result_cb(False, "goal_rejected")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_move_tcp_result)

    def _on_move_tcp_result(self, future):
        try:
            result_msg = future.result()
            result = result_msg.result
            if self._update_warn(not result.success, "move_tcp_result_failed", f"MoveTcp failed: {result.message}"):
                self._run_move_tcp_result_cb(False, result.message)
                return
            self._run_move_tcp_result_cb(True, result.message)
        except Exception as e:
            self._update_warn(True, "move_tcp_result_exception", f"MoveTcp result exception: {e}")
            self._run_move_tcp_result_cb(False, str(e))
            return
        self._update_warn(False, "move_tcp_result_exception")

    def _on_approach_done(self, success: bool, message: str):
        if self.current_status != STATUS_APPROACHING:
            return
        if self._update_warn(not success, "approach_failed", f"Approach move failed: {message}"):
            self._set_status(STATUS_PROMPTED)
            return
        self._set_status(STATUS_PROMPTED)

    def _run_move_tcp_result_cb(self, success: bool, message: str):
        cb = self._move_tcp_result_cb
        self._move_tcp_result_cb = None
        if cb is None:
            return
        try:
            cb(success, message)
        except Exception as e:
            self.get_logger().error(f"move_tcp result callback exception: {e}")

    def _make_base_pose_overlay(self, rgb, mask, K, p_obj_cam, centroid_uv):
        vis = rgb.copy()
        h, w = vis.shape[:2]
        img_cx = 0.5 * float(w - 1)
        img_cy = self.grasp_center_y * float(h - 1)
        center_px = (int(round(img_cx)), int(round(img_cy)))

        mask_u8 = (mask.astype(np.uint8) * 255)
        res = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = res[0] if len(res) == 2 else res[1]
        if contours:
            cv2.drawContours(vis, contours, -1, (255, 255, 0), 2)

        thr = max(0.0, float(self.grasp_center_dist_px_threshold))
        y_weight = max(0.0, float(self.grasp_center_y_weight))
        half_w = int(round(thr))
        half_h = int(round(thr / y_weight)) if y_weight > 1e-6 else h // 2

        x1 = max(0, min(w - 1, center_px[0] - half_w))
        y1 = max(0, min(h - 1, center_px[1] - half_h))
        x2 = max(0, min(w - 1, center_px[0] + half_w))
        y2 = max(0, min(h - 1, center_px[1] + half_h))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.circle(vis, center_px, 3, (255, 255, 255), -1)

        if centroid_uv is not None:
            cv2.circle(vis, centroid_uv, 5, (0, 255, 0), -1)
            cv2.line(vis, center_px, centroid_uv, (255, 255, 255), 1)

        if p_obj_cam is not None:
            uv = utils.project_uv(K, p_obj_cam)
            if uv is not None:
                cv2.circle(vis, uv, 5, (0, 255, 255), -1)

        return vis

    def _lookup_T_base_cam(self, cam_frame: str, stamp) -> Optional[np.ndarray]:
        timeout = Duration(seconds=self.tf_lookup_timeout_sec)
        try:
            t = Time.from_msg(stamp)
            tf = self.tf_buffer.lookup_transform(self.base_frame, cam_frame, t, timeout=timeout)
            self._update_warn(False, "tf_lookup_failed")
            self._update_warn(False, "tf_lookup_fallback_latest")
            return tf_to_T(tf)
        except Exception as e:
            try:
                tf_latest = self.tf_buffer.lookup_transform(self.base_frame, cam_frame, Time(), timeout=timeout)
                self._update_warn(
                    True,
                    "tf_lookup_fallback_latest",
                    f"TF timestamp lookup failed ({stamp.sec}.{stamp.nanosec}); using latest TF for "
                    f"{self.base_frame} <- {cam_frame}. reason: {e}",
                )
                self._update_warn(False, "tf_lookup_failed")
                return tf_to_T(tf_latest)
            except Exception as e2:
                self._update_warn(
                    True,
                    "tf_lookup_failed",
                    f"TF lookup failed: {self.base_frame} <- {cam_frame} at stamp={stamp.sec}.{stamp.nanosec}: {e2}",
                )
                return None

    def _on_synced(self, color_msg: Image, depth_msg: Image, mask_msg: Image):
        if not self._synced_once:
            self._synced_once = True
            self.get_logger().info("First synced color/depth/mask frame received.")

        if self._active_prompt_obj is None:
            if self.current_status != STATUS_READY:
                self._set_status(STATUS_READY)
            return

        if self.current_status in (STATUS_GRASPING, STATUS_APPROACHING):
            return

        if self._update_warn(self.ur5_status == "MOVING" and self.skip_while_moving, "ur5_moving", "UR5 is moving, skipping update"):
            return

        if self._update_warn(self.K is None, "no_camera_intrinsics", "Camera intrinsics not arrived"):
            self._set_status(STATUS_MISSING_INPUT)
            return

        if self._update_warn(
            self._required_target_mask_count is not None and self._target_mask_msg_count < self._required_target_mask_count,
            "waiting_target_mask",
            "Waiting for target mask",
        ):
            self._set_status(STATUS_MISSING_INPUT)
            return

        with self.lock:
            if self._update_warn(self.busy, "busy", "Busy"):
                return
            self.busy = True

        try:
            rgb = utils.decode_rgb(color_msg)
            depth = utils.decode_depth_m(depth_msg, self.depth_scale)
            inst = utils.decode_instance_mask(mask_msg)

            if depth.shape[:2] != rgb.shape[:2]:
                depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            if inst.shape[:2] != rgb.shape[:2]:
                inst = cv2.resize(inst, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

            cam_frame = self.camera_frame or (color_msg.header.frame_id or "")
            if self._update_warn(not cam_frame, "cam_frame_missing", "Camera frame missing"):
                self._set_status(STATUS_MISSING_INPUT)
                return

            mask = inst != 0
            if self._update_warn(int(mask.sum()) < self.min_mask_pixels, "mask_too_small", "Mask area too small"):
                self._set_status(STATUS_MISSING_INPUT)
                self.pose_tracker.reset_tracking()
                return

            if self._required_target_mask_count is not None:
                self._required_target_mask_count = None

            K = self.K.astype(np.float64, copy=False)
            depth = depth.astype(np.float32, copy=False)
            rgb = rgb.astype(np.uint8, copy=False)

            T_base_cam = self._lookup_T_base_cam(cam_frame, color_msg.header.stamp)
            if T_base_cam is None:
                self._set_status(STATUS_MISSING_INPUT)
                return

            if self.current_status == STATUS_MISSING_INPUT:
                self._set_status(STATUS_PROMPTED)

            p_obj_cam, z_mean, center_dist_px, centroid_uv = self._estimate_center_from_mask_depth(mask, depth, K)
            if self._update_warn(p_obj_cam is None, "mask_depth_invalid", "Mask/depth center estimation failed"):
                self._set_status(STATUS_MISSING_INPUT)
                return

            p_obj_base = (T_base_cam[:3, :3] @ p_obj_cam) + T_base_cam[:3, 3]
            vis_base = self._make_base_pose_overlay(rgb, mask, K, p_obj_cam, centroid_uv)
            vis = vis_base
            target_tcp = None

            if self._use_foundation_pose:
                T_cam_obj = self.pose_tracker.track_pose(K, rgb, depth, mask)
                if self._update_warn(T_cam_obj is None, "register_failed", "FoundationPose could not register target"):
                    self._set_status(STATUS_MISSING_INPUT)
                    return

                bad = self.pose_tracker.check_pose_valid(T_cam_obj, K, centroid_uv, z_mean)
                self._update_warn(bad, "tracking_unhealthy", "Tracking unhealthy, re-registering next frame")
                if bad:
                    self.pose_tracker.reset_tracking()

                T_base_obj = T_base_cam @ T_cam_obj
                self._publish_pose(self.pub_T_base_obj, self.base_frame, color_msg.header.stamp, T_base_obj)
                self._publish_pose(self.pub_T_cam_obj, cam_frame, color_msg.header.stamp, T_cam_obj)

                should_compute_target = not bad
                x_axis_base = None
                if should_compute_target:
                    target_tcp, x_axis_base = self.pose_tracker.compute_target_tcp_T(
                        T_base_obj=T_base_obj,
                        mask=mask,
                        K=K,
                        T_base_cam=T_base_cam,
                        z_ref_m=z_mean,
                        latest_tcp_R_base=self._latest_tcp_R_base,
                        latest_tcp_p_base=self._latest_tcp_p_base,
                        pregrasp_height=self.pregrasp_height,
                    )

                x_axis_cam = None
                if x_axis_base is not None:
                    x_axis_cam = T_base_cam[:3, :3].T @ x_axis_base

                vis = self.pose_tracker.overlay_pose_axes_and_center(vis_base, K, T_cam_obj, x_axis_cam)
            else:
                target_tcp = self._compute_target_tcp_from_point(p_obj_base)

            if target_tcp is not None:
                self._run_grasp_control(z_mean, center_dist_px, target_tcp, color_msg.header.stamp)
                self._log_grasp_gate(z_mean, center_dist_px)

            utils.publish_rgb8_image(self.pub_pose_vis, color_msg.header, vis)

        except Exception as e:
            self.get_logger().error(f"_on_synced exception: {e}\n{traceback.format_exc()}")
            self._set_status(STATUS_ERROR)
            self.pose_tracker.reset_tracking()
        finally:
            with self.lock:
                self.busy = False

def main():
    rclpy.init()
    node = ActorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
