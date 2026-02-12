#!/usr/bin/env python3
import sys
import threading
import traceback
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2

import tf2_ros
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger
from rfm.action import MoveTcp

import message_filters
from math_utils import mat_to_quat, quat_to_mat, tf_to_T
from publish_static_tf import StaticTFPublisher

# ----------------- PoseTracker -----------------
class PoseTrackerFoundationPose(Node):
    def __init__(self):
        super().__init__("pose_tracker_foundationpose")

        self.lock = threading.Lock()
        self.busy = False
        self.current_status = "READY"

        # -------- params --------
        repo_root = Path(__file__).resolve().parent
        self.repo_root = repo_root

        self.foundationpose_root =  str(repo_root / "thirdparty" / "FoundationPose")
        self.base_frame = self.declare_parameter("base_frame", "base").value
        self.camera_frame = self.declare_parameter("camera_frame", "camera_color_optical_frame").value  # empty -> use msg.header.frame_id

        self.depth_scale = 0.001  # uint16(mm)->m

        self.min_mask_pixels = int(self.declare_parameter("min_mask_pixels", 200).value)

        self.est_refine_iter = int(self.declare_parameter("est_refine_iter", 5).value)
        self.track_refine_iter = int(self.declare_parameter("track_refine_iter", 2).value)
        self.sync_queue_size = int(self.declare_parameter("sync_queue_size", 120).value)
        self.sync_slop_sec = float(self.declare_parameter("sync_slop_sec", 0.35).value)

        # TCP heuristic
        self.pregrasp_height = float(self.declare_parameter("pregrasp_height", 0.25).value)  # meters above object

        self.color_topic = "/wrist_cam/camera/color/image_raw"
        self.depth_topic = "/wrist_cam/camera/aligned_depth_to_color/image_raw"
        self.instance_mask_topic = "/perception/target_object_mask"
        self.initial_prompt = self.declare_parameter("prompt", "").value
        self.static_tf_file = str(self.declare_parameter("static_tf_file", str(self.repo_root / "T_wrist_cam.txt")).value)
        bridge_node_name = str(self.declare_parameter("bridge_node_name", "ur5").value).strip("/")
        self.bridge_prefix = f"/{bridge_node_name}"
        self.ur5_tcp_topic = self.declare_parameter("ur5_tcp_topic", f"{self.bridge_prefix}/tcp_pose").value
        self.ur5_status_topic = self.declare_parameter("ur5_status_topic", f"{self.bridge_prefix}/status").value
        self.ur5_move_tcp_action = self.declare_parameter("ur5_move_tcp_action", f"{self.bridge_prefix}/move_tcp").value
        self.ur5_action_wait_timeout = float(self.declare_parameter("ur5_action_wait_timeout", 0.05).value)

        # -------- TF --------
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.static_tf_publisher = StaticTFPublisher(self)
        try:
            self.static_tf_publisher.publish_from_file(matrix_file=self.static_tf_file, parent="tool0", child="camera_link")
        except Exception as e:
            self.get_logger().error(f"Failed to publish static TF from file '{self.static_tf_file}': {e}")

        # -------- outputs (relative to /pose_tracker namespace) --------
        self.pub_T_base_obj = self.create_publisher(PoseStamped, "/pose_tracker/T_base_obj", 10)
        self.pub_T_cam_obj = self.create_publisher(PoseStamped, "/pose_tracker/T_cam_obj", 10)
        self.pub_target_tcp = self.create_publisher(PoseStamped, "/pose_tracker/target_tcp_pose", 10)
        self.pub_pose_vis = self.create_publisher(Image, "/pose_tracker/pose_vis", 10)  # pose visualization (overlay on RGB)
        status_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_status = self.create_publisher(String, "/pose_tracker/status", status_qos)
        self.sub_tcp_pose = self.create_subscription(PoseStamped, self.ur5_tcp_topic, self._on_tcp_pose, 10)
        self._latest_tcp_R_base: Optional[np.ndarray] = None

        # UR5 status monitoring
        self.ur5_status = "IDLE"
        self.sub_ur5_status = self.create_subscription(String, self.ur5_status_topic, self._on_ur5_status, 10)
        self.act_move_tcp = ActionClient(self, MoveTcp, self.ur5_move_tcp_action)
        self._move_tcp_goal_pending = False

        # input: /inference/prompt  (typo fix)
        self.sub_prompt = self.create_subscription(String, "/inference/prompt", self._on_prompt, 10)

        # prompt/mask gating + debounce for sending tcp action goal
        self._has_prompt = False
        self._target_mask_msg_count = 0
        self._required_target_mask_count: Optional[int] = None
        self._good_start_ns = None  # nanoseconds; None when not in "good streak"
        self._good_required_ns = int(0.3 * 1e9)  # 0.3s
        self._warned_once_keys: set[str] = set()

        # service: /pose_tracker/reset
        self.srv_reset = self.create_service(Trigger, "/pose_tracker/reset", self._on_reset)

        # -------- camera intrinsics --------
        self.K: Optional[np.ndarray] = None
        self.sub_cam_info = self.create_subscription(CameraInfo, "/wrist_cam/camera/color/camera_info", self._on_cam_info, 10)
        self.sub_target_mask_arrived = self.create_subscription(
            Image, self.instance_mask_topic, self._on_target_mask_arrived, qos_profile_sensor_data
        )

        # -------- FoundationPose init --------
        if self.foundationpose_root not in sys.path:
            sys.path.insert(0, self.foundationpose_root)

        import torch
        import trimesh
        import nvdiffrast.torch as dr
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
        # Suppress noisy FoundationPose INFO logs (e.g. register/track internals).
        logging.getLogger().setLevel(logging.ERROR)

        self.torch = torch
        self.trimesh = trimesh
        self.FoundationPose = FoundationPose

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.fp_debug_dir = "/tmp/foundationpose_debug"

        # FoundationPose는 prompt 기반 객체가 정해질 때 lazy init/reset.
        self.mesh = None
        self.foundation_pose = None
        self.current_object_name = ""

        self.initialized = False
        self._status_timer = self.create_timer(0.3, self._publish_status)

        # -------- time sync: color + depth + instance_mask --------
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

        self.get_logger().info(
            "PoseTracker ready.\n"
            f"  Sub color: {self.color_topic}\n"
            f"  Sub depth: {self.depth_topic}\n"
            f"  Sub mask : {self.instance_mask_topic}\n"
            f"  Sub UR5 tcp/status: {self.ur5_tcp_topic}, {self.ur5_status_topic}\n"
            f"  Action MoveTcp: {self.ur5_move_tcp_action}\n"
            "  Pub /pose_tracker/T_base_obj, /pose_tracker/T_cam_obj, /pose_tracker/target_tcp_pose, /pose_tracker/status\n"
            f"  Sync queue/slop: {self.sync_queue_size}, {self.sync_slop_sec:.3f}s\n"
        )

        if self.initial_prompt:
            self._on_prompt(String(data=self.initial_prompt))
        self._publish_status()

    def _normalize_object_name(self, name: str) -> str:
        return (name or "").strip().lower().replace(" ", "_")

    def _resolve_mesh_file(self, object_name: str):
        norm = self._normalize_object_name(object_name)
        if not norm:
            return None, None

        ycb_root = self.repo_root / "ycb"
        if not ycb_root.exists():
            self.get_logger().error(f"ycb directory not found: {ycb_root}")
            return None, None

        pattern = f"*{norm}*"
        matches = sorted(p for p in ycb_root.glob(pattern) if p.is_dir())
        if not matches:
            self.get_logger().error(f"No YCB directory matched object '{object_name}' under {ycb_root}")
            return None, None

        mesh_file = matches[0] / "google_16k" / "textured.obj"
        if not mesh_file.exists():
            self.get_logger().error(f"Mesh file not found: {mesh_file}")
            return None, None

        self.get_logger().info(f"Loading mesh for '{object_name}': {mesh_file}")
        mesh = self.trimesh.load(str(mesh_file))
        mesh.vertices = mesh.vertices.astype(np.float64)
        if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None:
            mesh.vertex_normals = mesh.vertex_normals.astype(np.float64)

        model_normals = mesh.vertex_normals if getattr(mesh, "vertex_normals", None) is not None else np.zeros_like(mesh.vertices)

        return mesh, model_normals

    def _reset_foundation_pose_object(self, object_name: str) -> bool:
        mesh, model_normals = self._resolve_mesh_file(object_name)
        if mesh is None:
            return False

        try:
            if self.foundation_pose is None:
                self.foundation_pose = self.FoundationPose(
                    model_pts=mesh.vertices,
                    model_normals=model_normals,
                    mesh=mesh,
                    scorer=self.scorer,
                    refiner=self.refiner,
                    debug_dir=self.fp_debug_dir,
                    debug=0,
                    glctx=self.glctx,
                )
            else:
                self.foundation_pose.reset_object(
                    model_pts=mesh.vertices,
                    model_normals=model_normals,
                    symmetry_tfs=None,
                    mesh=mesh,
                )
            # --- Analyze Mesh for Grasping ---
            try:
                # trimesh.bounds.oriented_bounds(mesh) returns (transform, extents)
                self.T_obb_mesh, self.obb_extents = self.trimesh.bounds.oriented_bounds(mesh)
            except Exception as e:
                self.get_logger().error(f"Failed to compute OBB: {e}")
                self.T_obb_mesh, self.obb_extents = np.eye(4), np.array([0.1, 0.1, 0.1])
            self.idx_shortest = np.argmin(self.obb_extents)
            
            # Compute Mesh Centroid (Min/Max based)
            min_bound = mesh.vertices.min(axis=0)
            max_bound = mesh.vertices.max(axis=0)
            self.mesh_centroid_obj = (min_bound + max_bound) / 2.0
            
            self.get_logger().info(f"Loaded '{object_name}': OBB extents={self.obb_extents}, shortest_idx={self.idx_shortest}, centroid={self.mesh_centroid_obj}")
            # ---------------------------------

            self.mesh = mesh
            self.current_object_name = self._normalize_object_name(object_name)
            self.initialized = False
            self._set_status("READY")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to reset FoundationPose mesh for '{object_name}': {e}")
            return False

    def _extract_object_from_prompt(self, prompt_text: str) -> Optional[str]:
        parts = (prompt_text or "").strip().split()
        if len(parts) < 2:
            return None
        return " ".join(parts[1:]).strip()

    def _on_prompt(self, msg: String):
        obj_name = self._extract_object_from_prompt(msg.data)
        if obj_name is None:
            self.get_logger().warning(f"Invalid prompt format: '{msg.data}'. Expected: '<verb> <object>'")
            return
        ok = self._reset_foundation_pose_object(obj_name)
        if not ok:
            self._set_status("ERROR")
            self._has_prompt = False
            self._required_target_mask_count = None
            return
        self._has_prompt = True
        self._required_target_mask_count = self._target_mask_msg_count + 1
        self._good_start_ns = None  # prompt 바뀌면 안정화 다시

    def _publish_status(self):
        msg = String()
        msg.data = self.current_status
        self.pub_status.publish(msg)

    def _set_status(self, s: str):
        if s != self.current_status:
            self.current_status = s
            self._publish_status()

    def _warn_once(self, key: str, msg: str):
        if key in self._warned_once_keys:
            return
        self._warned_once_keys.add(key)
        self.get_logger().warning(msg)

    def _clear_warn(self, key: str):
        self._warned_once_keys.discard(key)


    def _on_reset(self, req, resp):
        self.initialized = False
        self._set_status("READY")
        self._good_start_ns = None
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
        if self.ur5_status == "IDLE":
            self._move_tcp_goal_pending = False

    def _on_target_mask_arrived(self, _msg: Image):
        self._target_mask_msg_count += 1

    def _decode_rgb(self, msg: Image) -> np.ndarray:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        enc = msg.encoding.lower()

        if enc in ("rgb8", "rgba8"):
            rgb = img[:, :, :3]
            return rgb
        if enc in ("bgr8", "bgra8"):
            bgr = img[:, :, :3]
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        raise RuntimeError(f"Unsupported color encoding: {msg.encoding}")

    def _decode_depth_m(self, msg: Image) -> np.ndarray:
        enc = msg.encoding.lower()
        if enc in ("16uc1", "mono16"):
            d = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            depth = d.astype(np.float32) * self.depth_scale
            return depth
        if enc in ("32fc1",):
            d = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            d[~np.isfinite(d)] = 0.0
            return d
        raise RuntimeError(f"Unsupported depth encoding: {msg.encoding}")
    
    def _decode_instance_mask(self, msg: Image) -> np.ndarray:
        enc = msg.encoding.lower()
        if enc in ("32sc1",):
            m = np.frombuffer(msg.data, dtype=np.int32).reshape(msg.height, msg.width)
            return m
        if enc in ("mono8", "8uc1"):
            m = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width).astype(np.int32)
            return m
        raise RuntimeError(f"Unsupported mask encoding: {msg.encoding}")

    def _compute_target_tcp_T(self, T_base_obj: np.ndarray, x_axis_base: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        # T_base_obj: 4x4 (Object Pose from FoundationPose)
        # x_axis_base: 3x1 (Shortest horizontal axis of the object in Base Frame)
        # Note: FoundationPose T_base_obj origin is usually arbitrary (mesh origin).
        
        # 0. Correct Object Position (Centroid)
        # Transform mesh centroid to Base Frame
        if hasattr(self, 'mesh_centroid_obj'):
            p_center_obj = self.mesh_centroid_obj
            # p_center_base = R * p_center_obj + t
            p_obj_base = (T_base_obj[:3, :3] @ p_center_obj) + T_base_obj[:3, 3]
        else:
            # Fallback
            p_obj_base = T_base_obj[:3, 3]

        # Check if we have OBB info
        if not hasattr(self, 'T_obb_mesh') or self.T_obb_mesh is None:
            # Fallback to old behavior
            if self._latest_tcp_R_base is None:
                self._warn_once("tcp_pose_missing", f"Current TCP pose is not available yet ({self.ur5_tcp_topic}).")
                return None
            R_tcp = self._latest_tcp_R_base
            z_tcp_base = R_tcp[:, 2]
            z_tcp_base /= (np.linalg.norm(z_tcp_base) + 1e-12)
            p_tcp = p_obj_base - z_tcp_base * self.pregrasp_height
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_tcp
            T[:3, 3] = p_tcp
            return T

        # 1. Determine Approach Axis (Target Gripper Z)
        # "Dynamic Approach": Point Z-axis from Current TCP -> Object Centroid
        if not hasattr(self, '_latest_tcp_p_base') or self._latest_tcp_p_base is None:
            self._warn_once("tcp_pose_missing", f"Current TCP pose is not available yet ({self.ur5_tcp_topic}).")
            return None

        p_current = self._latest_tcp_p_base
        v_approach = p_obj_base - p_current
        dist = np.linalg.norm(v_approach)
        if dist > 1e-3:
            v_approach /= dist
        else:
            v_approach = np.array([0.0, 0.0, -1.0]) # Fallback (at object?)

        # Project ideal approach to be perpendicular to x_target
        dot_val = np.dot(v_approach, x_axis_base)
        z_raw = v_approach - dot_val * (x_axis_base)
        
        if np.linalg.norm(z_raw) < 0.1:
             # ideal approach is parallel to X
             z_target = np.array([0.0, 0.0, -1.0])
        else:
             z_target = z_raw / np.linalg.norm(z_raw)

        # 2. Compute Gripper Y
        y_target = np.cross(z_target, x_axis_base) 
        
        # Construct R
        R_tcp = np.column_stack((x_axis_base, y_target, z_target))
        
        # 3. Compute Position
        # Apply pregrasp offset along wrist-local +Z, converted into base frame.
        z_wrist_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        z_wrist_in_base = R_tcp @ z_wrist_local
        p_tcp = p_obj_base - z_wrist_in_base * self.pregrasp_height
        
        # 4. Safety Checks
        if p_tcp[2] < 0.2:
            self._warn_once("tcp_too_low", f"Computed TCP z={p_tcp[2]:.3f} is too low. Clamping.")
            p_tcp[2] = 0.2
        self._clear_warn("tcp_too_low")

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_tcp
        T[:3, 3] = p_tcp
        return T

    @staticmethod
    def _make_pose_msg(frame_id: str, stamp, T: np.ndarray) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp

        q = mat_to_quat(T[:3, :3])
        t = T[:3, 3]

        msg.pose.position.x = float(t[0])
        msg.pose.position.y = float(t[1])
        msg.pose.position.z = float(t[2])
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        return msg

    def _publish_pose(self, topic_pub, frame_id: str, stamp, T: np.ndarray):
        msg = self._make_pose_msg(frame_id, stamp, T)
        topic_pub.publish(msg)

    def _send_move_tcp_goal(self, T_base_tcp: np.ndarray, stamp):
        if self._move_tcp_goal_pending:
            self._warn_once("move_tcp_pending", "MoveTcp goal is still pending. Skip new goal.")
            return
        self._clear_warn("move_tcp_pending")
        if not self.act_move_tcp.wait_for_server(timeout_sec=self.ur5_action_wait_timeout):
            self._warn_once("move_tcp_server_unavailable", f"MoveTcp action server unavailable: {self.ur5_move_tcp_action}")
            return
        self._clear_warn("move_tcp_server_unavailable")

        goal = MoveTcp.Goal()
        goal.relative = False
        goal.target_pose = self._make_pose_msg(self.base_frame, stamp, T_base_tcp)

        self._move_tcp_goal_pending = True
        send_future = self.act_move_tcp.send_goal_async(goal)
        send_future.add_done_callback(self._on_move_tcp_goal_response)

    def _on_move_tcp_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self._move_tcp_goal_pending = False
            self._warn_once("move_tcp_send_failed", f"MoveTcp goal send failed: {e}")
            return
        self._clear_warn("move_tcp_send_failed")

        if not goal_handle.accepted:
            self._move_tcp_goal_pending = False
            self._warn_once("move_tcp_goal_rejected", "MoveTcp goal rejected by action server.")
            return
        self._clear_warn("move_tcp_goal_rejected")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_move_tcp_result)

    def _on_move_tcp_result(self, future):
        self._move_tcp_goal_pending = False
        try:
            result_msg = future.result()
            result = result_msg.result
            if not result.success:
                self._warn_once("move_tcp_result_failed", f"MoveTcp failed: {result.message}")
                return
            self._clear_warn("move_tcp_result_failed")
        except Exception as e:
            self._warn_once("move_tcp_result_exception", f"MoveTcp result exception: {e}")
            return
        self._clear_warn("move_tcp_result_exception")

    @staticmethod
    def _project_uv(K: np.ndarray, p_cam: np.ndarray) -> Optional[Tuple[int, int]]:
        """Project 3D point in camera coords -> pixel (u,v). Return None if behind camera."""
        z = float(p_cam[2])
        if (not np.isfinite(z)) or z <= 1e-6:
            return None
        uvw = K @ p_cam.reshape(3,)
        w = float(uvw[2])
        if (not np.isfinite(w)) or w <= 1e-6:
            return None
        u = float(uvw[0] / w)
        v = float(uvw[1] / w)
        return int(round(u)), int(round(v))

    def _publish_rgb8_image(self, pub, header, rgb: np.ndarray):
        if rgb is None:
            return
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8, copy=False)

        msg = Image()
        msg.header = header
        msg.height = int(rgb.shape[0])
        msg.width = int(rgb.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = int(msg.width * 3)
        msg.data = rgb.tobytes()
        pub.publish(msg)

    def _make_pose_vis(self, rgb: np.ndarray, mask: np.ndarray, K: np.ndarray, T_cam_obj: np.ndarray, x_axis_cam: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw mask contour + xyz axes (R,G,B) from T_cam_obj on an RGB image."""
        vis = rgb.copy()

        # draw contour (yellow)
        if mask is not None:
            mask_u8 = (mask.astype(np.uint8) * 255)
            res = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = res[0] if len(res) == 2 else res[1]
            if contours:
                cv2.drawContours(vis, contours, -1, (255, 255, 0), 2)  # RGB yellow

        # axes
        R = T_cam_obj[:3, :3]
        t = T_cam_obj[:3, 3]
        L = 0.05 # Visualized axis length (in real world scale, 단위 : m)

        pts_obj = np.array([
            [0.0, 0.0, 0.0],  # origin
            [L,   0.0, 0.0],  # x
            [0.0, L,   0.0],  # y
            [0.0, 0.0, L  ],  # z
        ], dtype=np.float64)

        pts_cam = (R @ pts_obj.T).T + t.reshape(1, 3)

        p0 = self._project_uv(K, pts_cam[0])
        px = self._project_uv(K, pts_cam[1])
        py = self._project_uv(K, pts_cam[2])
        pz = self._project_uv(K, pts_cam[3])

        th = 3
        if p0 is not None:
            cv2.circle(vis, p0, max(2, th + 1), (255, 255, 255), -1)  # white

        # Draw Centroid (Cyan)
        p_center_uv = None
        if hasattr(self, 'mesh_centroid_obj'):
            p_center_obj = self.mesh_centroid_obj
            p_center_cam = (T_cam_obj[:3, :3] @ p_center_obj) + T_cam_obj[:3, 3]
            p_center_uv = self._project_uv(K, p_center_cam)
            if p_center_uv is not None:
                cv2.circle(vis, p_center_uv, 5, (0, 255, 255), -1)

        # Draw Shortest Axis (Magenta)
        # x_axis_cam is a Direction vector in Camera Frame. 
        # Start at Centroid (if available) or Origin.
        if x_axis_cam is not None:
            # Using Centroid as start if available
            start_cam = p_center_cam if (hasattr(self, 'mesh_centroid_obj') and 'p_center_cam' in locals()) else t
            end_cam = start_cam + x_axis_cam * 0.08 # 8cm length
            
            uv_start = self._project_uv(K, start_cam)
            uv_end = self._project_uv(K, end_cam)
            
            if uv_start is not None and uv_end is not None:
                cv2.line(vis, uv_start, uv_end, (255, 0, 255), 2) # Magenta

        # X=red, Y=green, Z=blue (RGB)
        if p0 is not None and px is not None:
            cv2.line(vis, p0, px, (255, 0, 0), th)
        if p0 is not None and py is not None:
            cv2.line(vis, p0, py, (0, 255, 0), th)
        if p0 is not None and pz is not None:
            cv2.line(vis, p0, pz, (0, 0, 255), th)
        return vis

    def _broadcast_object_tf(self, obj_id: int, frame_id: str, stamp, T: np.ndarray):
        tf = TransformStamped()
        tf.header.frame_id = frame_id
        tf.header.stamp = stamp
        tf.child_frame_id = f"object_{obj_id}"

        q = mat_to_quat(T[:3, :3])
        t = T[:3, 3]
        tf.transform.translation.x = float(t[0])
        tf.transform.translation.y = float(t[1])
        tf.transform.translation.z = float(t[2])
        tf.transform.rotation.x = float(q[0])
        tf.transform.rotation.y = float(q[1])
        tf.transform.rotation.z = float(q[2])
        tf.transform.rotation.w = float(q[3])
        self.tf_broadcaster.sendTransform(tf)

    def _lookup_T_base_cam(self, cam_frame: str, stamp) -> Optional[np.ndarray]:
        try:
            t = Time.from_msg(stamp)
            tf = self.tf_buffer.lookup_transform(self.base_frame, cam_frame, t, timeout=Duration(seconds=0.1))
            self._clear_warn("tf_lookup_failed")
            return tf_to_T(tf)
        except Exception as e:
            self._warn_once("tf_lookup_failed",
                f"TF lookup failed: {self.base_frame} <- {cam_frame} at stamp={stamp.sec}.{stamp.nanosec}: {e}"
            )
            return None

    def _on_synced(self, color_msg: Image, depth_msg: Image, mask_msg: Image):
        if not self._synced_once:
            self._synced_once = True
            self.get_logger().info("First synced color/depth/mask frame received.")
        # UR5 Moving Check
        if self.ur5_status == "MOVING":
            self._warn_once("ur5_moving", "UR5 is Moving, skipping pose estimation")
            return
        self._clear_warn("ur5_moving")
        
        if self.K is None:
            self._warn_once("no_camera_intrinsics", "Camera Intrinsics not arrived")
            return
        self._clear_warn("no_camera_intrinsics")

        if self.mesh is None:
            self._warn_once("mesh_not_loaded", "Mesh not Loaded")
            self._set_status("READY")
            return
        self._clear_warn("mesh_not_loaded")

        # /inference/prompt와 해당 prompt 이후 target mask가 모두 준비되어야 진행
        if not self._has_prompt:
            self._warn_once("prompt_not_arrived", "Prompt has not arrived")
            return
        self._clear_warn("prompt_not_arrived")

        if (self._required_target_mask_count is not None) and self._target_mask_msg_count < self._required_target_mask_count:
            self._warn_once("waiting_target_mask", "Waiting for Target Mask")
            self._set_status("WAIT_MASK")
            return
        self._clear_warn("waiting_target_mask")

        with self.lock:
            if self.busy:
                self._warn_once("busy", "Busy")
                return
            self.busy = True
        self._clear_warn("busy")

        try:
            rgb = self._decode_rgb(color_msg)
            depth = self._decode_depth_m(depth_msg)
            inst = self._decode_instance_mask(mask_msg)

            # size safety
            if depth.shape[:2] != rgb.shape[:2]:
                depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            if inst.shape[:2] != rgb.shape[:2]:
                inst = cv2.resize(inst, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

            # camera frame
            cam_frame = self.camera_frame or (color_msg.header.frame_id or "")
            if not cam_frame:
                self._warn_once("cam_frame_missing", "Cam Frame has not arrived")
                self._set_status("ERROR")
                return
            self._clear_warn("cam_frame_missing")

            # target_object_mask는 0/1(또는 0/nonzero) 단일 타겟 마스크로 가정
            mask = (inst != 0)

            if int(mask.sum()) < self.min_mask_pixels:
                self._warn_once("mask_too_small", "Mask Area too small")
                self._set_status("NO_TARGET")
                self.initialized = False
                self._good_start_ns = None
                return
            self._clear_warn("mask_too_small")
            if self._required_target_mask_count is not None:
                self._required_target_mask_count = None
            K = self.K.astype(np.float64, copy=False)
            depth = depth.astype(np.float32, copy=False)
            rgb = rgb.astype(np.uint8, copy=False) 

            # --- FoundationPose ---
            with self.torch.inference_mode():
                if not self.initialized:
                    # register once using mask
                    T_cam_obj = self.foundation_pose.register(K=K, rgb=rgb, depth=depth,
                                                      ob_mask=mask, iteration=self.est_refine_iter)
                    if T_cam_obj is None:
                        self._warn_once("register_failed", "FoundationPose could not register target")
                        self._set_status("NO_TARGET")
                        return
                    self._clear_warn("register_failed")
                    self.initialized = True
                else:
                    T_cam_obj = self.foundation_pose.track_one(rgb=rgb, depth=depth, K=K,
                                                   iteration=self.track_refine_iter)

            T_cam_obj = np.asarray(T_cam_obj, dtype=np.float64)
            if T_cam_obj.shape != (4, 4):
                self._set_status("ERROR")
                self.initialized = False
                return
            bad = self.debug_T_cam_obj(depth, mask, T_cam_obj, K)
                
            if bad:
                self.initialized = False
                self._set_status("READY")
            else:
                self._clear_warn("tracking_unhealthy")

            # --- TF: base <- camera ---
            T_base_cam = self._lookup_T_base_cam(cam_frame, color_msg.header.stamp)
            if T_base_cam is None:
                self._set_status("ERROR")
                return

            T_base_obj = T_base_cam @ T_cam_obj

            self._publish_pose(self.pub_T_base_obj, self.base_frame, color_msg.header.stamp, T_base_obj)
            self._publish_pose(self.pub_T_cam_obj, cam_frame, color_msg.header.stamp, T_cam_obj)

            # optional TF object_<id> (단일 타겟이므로 id=1)
            self._broadcast_object_tf(1, self.base_frame, color_msg.header.stamp, T_base_obj)

            # Compute Target TCP
            now_ns = int(self.get_clock().now().nanoseconds)
            
            # Helper: Compute Horizontal X-Axis for Vis & Control
            x_axis_base = self._compute_horizontal_x_axis(T_base_obj)

            if bad:
                self._good_start_ns = None
            else:
                if self._good_start_ns is None:
                    self._good_start_ns = now_ns
                if (now_ns - self._good_start_ns) >= self._good_required_ns:
                    # Use the pre-computed x_axis_base
                    target_tcp = self._compute_target_tcp_T(T_base_obj, x_axis_base)
                    
                    if target_tcp is None:
                        self._set_status("WAIT_TCP")
                        return
                    self._clear_warn("tcp_pose_missing")
                    self._publish_pose(self.pub_target_tcp, self.base_frame, color_msg.header.stamp, target_tcp)
                    self._send_move_tcp_goal(target_tcp, color_msg.header.stamp)

            # Visualize Estimated object 6d pos + Centroid + Shortest Axis
            # Transform x_axis_base to Camera Frame for visualization
            x_axis_cam = None
            if x_axis_base is not None and T_base_cam is not None:
                R_base_cam = T_base_cam[:3, :3]
                # x_cam = R_cam_base @ x_base = R_base_cam.T @ x_base
                x_axis_cam = R_base_cam.T @ x_axis_base

            vis = self._make_pose_vis(rgb, mask, K, T_cam_obj, x_axis_cam)
            self._publish_rgb8_image(self.pub_pose_vis, color_msg.header, vis)

            self._set_status("RUNNING")
        except Exception as e:
            self.get_logger().error(f"_on_synced exception: {e}\n{traceback.format_exc()}")
            self._set_status("ERROR")
            self.initialized = False
            self._good_start_ns = None
        finally:
            with self.lock:
                self.busy = False
    
    def _compute_horizontal_x_axis(self, T_base_obj: np.ndarray) -> Optional[np.ndarray]:
        """Compute the shortest horizontal axis of the object in Base Frame."""
        if not hasattr(self, 'T_obb_mesh') or self.T_obb_mesh is None:
            return None

        R_base_obj = T_base_obj[:3, :3]
        R_obb_mesh = self.T_obb_mesh[:3, :3]
        extents = self.obb_extents
        
        # T_obb_mesh is Transform from Mesh to OBB (Aligning). 
        # So to get OBB axes in Mesh Frame, we need the inverse rotation.
        # R_obb_mesh is Mesh->OBB. So R_obb_mesh.T is OBB->Mesh.
        # The axes of OBB (standard basis) in Mesh Frame are the columns of R_obb_mesh.T,
        # which are the rows of R_obb_mesh.
        axes_mesh = [R_obb_mesh[0, :], R_obb_mesh[1, :], R_obb_mesh[2, :]]
        sorted_indices = np.argsort(extents)
        
        best_x_target = None
        
        for idx in sorted_indices:
            v_axis_mesh = axes_mesh[idx]
            v_axis_base = R_base_obj @ v_axis_mesh
            v_axis_base /= (np.linalg.norm(v_axis_base) + 1e-9)
            if abs(v_axis_base[2]) < 0.7:
                best_x_target = v_axis_base
                break
        
        if best_x_target is None:
            v_axis_base = R_base_obj @ axes_mesh[sorted_indices[0]]
            v_axis_base[2] = 0
            best_x_target = v_axis_base
            
        best_x_target[2] = 0.0
        x_target = best_x_target / (np.linalg.norm(best_x_target) + 1e-9)

        # Orientation Continuity
        if hasattr(self, '_latest_tcp_R_base') and self._latest_tcp_R_base is not None:
            current_x = self._latest_tcp_R_base[:, 0]
            if np.dot(x_target, current_x) < 0:
                x_target = -x_target
        
        return x_target

    def debug_T_cam_obj(self, depth, mask, T_cam_obj, K):
        # FoundationPose가 계산한 camera 기준 object의 위치(T_cam_obj)가 정확한지 디버깅
        # T_cam_obj의 xy 값은 Cutie가 찍은 mask centroid와 비교
        # T_cam_obj의 z 값은 realsense depth sensor가 찍은 값과 비교
        z_med = np.median(depth[mask])
        T_cam_obj_z = T_cam_obj[2, 3]
        
        # 1) valid-depth mask (exclude zeros + optionally erode to avoid boundary holes)
        mask_u8 = mask.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask_er = cv2.erode(mask_u8, kernel, iterations=1).astype(bool)

        depth_roi = depth[mask_er]
        valid_depth = depth_roi[np.isfinite(depth_roi) & (depth_roi > 1e-6)]
        valid_ratio = float(valid_depth.size) / float(mask_er.sum() + 1e-12)

        if valid_depth.size == 0:
            z_med_valid = float("nan")
        else:
            z_med_valid = float(np.median(valid_depth))

        # 2) projection error (mask center vs projected pose origin)
        ys, xs = np.nonzero(mask_er)
        u_mask = float(xs.mean()) if xs.size else float("nan")
        v_mask = float(ys.mean()) if ys.size else float("nan")
        uv_pose = self._project_uv(K, T_cam_obj[:3, 3].astype(np.float64, copy=False))
        if uv_pose is None or not np.isfinite(u_mask) or not np.isfinite(v_mask):
            dist_px = float("nan")
        else:
            u_pose, v_pose = uv_pose
            dist_px = float(np.hypot(u_pose - u_mask, v_pose - v_mask))

        # 3) health check -> tracking reset trigget
        # - valid depth가 너무 적거나
        # - z가 depth median과 크게 어긋나거나
        # - pose origin projection이 mask 중심에서 너무 멀면
        # - bad = True
        bad = False
        if (not np.isfinite(z_med_valid)) or (valid_ratio < 0.15):
            bad = True
        if np.isfinite(z_med_valid) and abs(z_med_valid - T_cam_obj_z) > 0.1:
            bad = True
        if np.isfinite(dist_px) and dist_px > 60.0:
            bad = True

        if bad:
            self._warn_once("tracking_unhealthy",
                "diag: tracking unhealthy -> force re-register next frame "
                f"(valid_ratio={valid_ratio:.2f}, z_diff={(z_med_valid - T_cam_obj_z):.3f}, "
                f"dist_px={dist_px:.1f}"
            )

        return bad

def main():
    rclpy.init()
    node = PoseTrackerFoundationPose()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
