#!/usr/bin/env python3
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2

import tf2_ros
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import String, Int32
from std_srvs.srv import Trigger

import message_filters
from math_utils import quat_to_mat, mat_to_quat, tf_to_T, pose_to_T



# ----------------- PoseTracker -----------------
class PoseTrackerFoundationPose(Node):
    def __init__(self):
        super().__init__("pose_tracker_foundationpose")

        self.lock = threading.Lock()
        self.busy = False

        # -------- params --------
        repo_root = Path(__file__).resolve().parent
        self.repo_root = repo_root

        self.foundationpose_root = self.declare_parameter(
            "foundationpose_root", str(repo_root / "thirdparty" / "FoundationPose")
        ).value
        self.base_frame = self.declare_parameter("base_frame", "base").value
        self.camera_frame = self.declare_parameter("camera_frame", "camera_color_optical_frame").value  # empty -> use msg.header.frame_id

        self.depth_scale = float(self.declare_parameter("depth_scale", 0.001).value)  # uint16(mm)->m

        self.target_object_id = int(self.declare_parameter("target_object_id", 0).value)  # 0 => auto(largest)
        self.min_mask_pixels = int(self.declare_parameter("min_mask_pixels", 200).value)

        self.est_refine_iter = int(self.declare_parameter("est_refine_iter", 5).value)
        self.track_refine_iter = int(self.declare_parameter("track_refine_iter", 2).value)

        # TCP heuristic
        self.pregrasp_height = float(self.declare_parameter("pregrasp_height", 0.10).value)  # meters above object

        self.publish_object_tf = bool(self.declare_parameter("publish_object_tf", True).value)
        
        # pose visualization (overlay on RGB)
        self.publish_pose_vis = bool(self.declare_parameter("publish_pose_vis", True).value)
        self.pose_vis_axis_len = float(self.declare_parameter("pose_vis_axis_len", 0.05).value)  # meters
        self.pose_vis_thickness = int(self.declare_parameter("pose_vis_thickness", 3).value)
        self.pose_vis_draw_contour = bool(self.declare_parameter("pose_vis_draw_contour", True).value)
        self.pub_pose_vis = self.create_publisher(Image, "pose_vis", 10)

        # topics (absolute defaults as spec)
        self.color_topic = self.declare_parameter(
            "color_topic", "/wrist_cam/camera/color/image_raw"
        ).value
        self.depth_topic = self.declare_parameter(
            "depth_topic", "/wrist_cam/camera/aligned_depth_to_color/image_raw"
        ).value
        self.cam_info_topic = self.declare_parameter(
            "camera_info_topic", "/wrist_cam/camera/color/camera_info"
        ).value
        self.instance_mask_topic = self.declare_parameter(
            "instance_mask_topic", "/perception/instance_mask"
        ).value
        self.prompt_topic = self.declare_parameter("prompt_topic", "/inference/prompt").value
        self.initial_prompt = self.declare_parameter("prompt", "").value

        # -------- TF --------
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # -------- outputs (relative to /pose_tracker namespace) --------
        self.pub_T_base_obj = self.create_publisher(PoseStamped, "T_base_obj", 10)
        self.pub_T_cam_obj = self.create_publisher(PoseStamped, "T_cam_obj", 10)
        self.pub_target_tcp = self.create_publisher(PoseStamped, "goal_tcp_pose", 10)
        self.pub_status = self.create_publisher(String, "status", 10)

        # optional input: /pose_tracker/target_object_id
        self.sub_target_id = self.create_subscription(Int32, "target_object_id", self._on_target_id, 10)
        self.sub_prompt = self.create_subscription(String, self.prompt_topic, self._on_prompt, 10)

        # service: /pose_tracker/reset
        self.srv_reset = self.create_service(Trigger, "reset", self._on_reset)

        # -------- camera intrinsics --------
        self.K: Optional[np.ndarray] = None
        self._cam_info_sub = self.create_subscription(CameraInfo, self.cam_info_topic, self._on_cam_info, 10)

        # -------- FoundationPose init --------
        if self.foundationpose_root not in sys.path:
            sys.path.insert(0, self.foundationpose_root)
        os.chdir(self.foundationpose_root)
        
        import torch
        import trimesh
        import nvdiffrast.torch as dr
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

        self.torch = torch

        self.trimesh = trimesh

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()

        # Load Default Mesh 
        self.mesh, model_normals = self._resolve_mesh_file('apple')
        self.foundation_pose = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=model_normals,
            mesh=self.mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir="/tmp/foundationpose_debug",
            debug=0,
            glctx=self.glctx,
        )
        self.current_object_name = ""

        self.initialized = False
        self.current_status = "READY"
        self._status_timer = self.create_timer(0.3, self._publish_status)

        # -------- time sync: color + depth + instance_mask --------
        color_sub = message_filters.Subscriber(
            self, Image, self.color_topic, qos_profile=qos_profile_sensor_data
        )
        depth_sub = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data
        )
        mask_sub = message_filters.Subscriber(
            self, Image, self.instance_mask_topic, qos_profile=qos_profile_sensor_data
        )

        sync = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, mask_sub], 10, 0.05)
        sync.registerCallback(self._on_synced)

        self.get_logger().info(
            "PoseTracker ready.\n"
            f"  Sub color: {self.color_topic}\n"
            f"  Sub depth: {self.depth_topic}\n"
            f"  Sub mask : {self.instance_mask_topic}\n"
            f"  Pub /pose_tracker/T_base_obj, /pose_tracker/T_cam_obj, /pose_tracker/goal_tcp_pose, /pose_tracker/status\n"
        )

        if self.initial_prompt:
            self._on_prompt(String(data=self.initial_prompt))

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
            self.foundation_pose.reset_object(
                model_pts=mesh.vertices,
                model_normals=model_normals,
                symmetry_tfs=None,
                mesh=mesh,
            )
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
        return parts[1]

    def _on_prompt(self, msg: String):
        obj_name = self._extract_object_from_prompt(msg.data)
        if obj_name is None:
            self.get_logger().warning(f"Invalid prompt format: '{msg.data}'. Expected: '<verb> <object>'")
            return
        ok = self._reset_foundation_pose_object(obj_name)
        if not ok:
            self._set_status("ERROR")

    def _publish_status(self):
        msg = String()
        msg.data = self.current_status
        self.pub_status.publish(msg)

    def _set_status(self, s: str):
        if s != self.current_status:
            self.current_status = s

    def _on_target_id(self, msg: Int32):
        self.target_object_id = int(msg.data)
        # 새 target id 지정되면 다음 프레임에서 register 다시 하도록
        self.initialized = False
        self._set_status("READY")

    def _on_reset(self, req, resp):
        self.initialized = False
        self._set_status("READY")
        resp.success = True
        resp.message = "reset ok"
        return resp

    def _on_cam_info(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

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

    def _choose_target_id(self, inst: np.ndarray) -> int:
        # 0 => background, 1..N => track_id
        ids, cnt = np.unique(inst, return_counts=True)
        valid = [(i, c) for i, c in zip(ids.tolist(), cnt.tolist()) if i != 0]
        if not valid:
            return 0
        valid.sort(key=lambda x: x[1], reverse=True)
        return int(valid[0][0])

    def _compute_target_tcp_T(self, T_base_obj: np.ndarray) -> np.ndarray:
        p = T_base_obj[:3, 3]
        R_obj = T_base_obj[:3, :3]

        # z down (base frame)
        z_tcp = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        # align x with object x projected on xy
        x_obj = R_obj[:, 0].copy()
        x_obj[2] = 0.0
        n = np.linalg.norm(x_obj)
        if n < 1e-6:
            x_tcp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            x_tcp = x_obj / n

        y_tcp = np.cross(z_tcp, x_tcp)
        y_tcp /= (np.linalg.norm(y_tcp) + 1e-12)
        x_tcp = np.cross(y_tcp, z_tcp)  # re-orthogonalize

        R_tcp = np.stack([x_tcp, y_tcp, z_tcp], axis=1)

        p_tcp = p - z_tcp * self.pregrasp_height  # up by pregrasp_height
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_tcp
        T[:3, 3] = p_tcp
        return T

    def _publish_pose(self, topic_pub, frame_id: str, stamp, T: np.ndarray):
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

        topic_pub.publish(msg)

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

    def _make_pose_vis(self, rgb: np.ndarray, mask: np.ndarray, K: np.ndarray, T_cam_obj: np.ndarray) -> np.ndarray:
        """Draw mask contour + xyz axes (R,G,B) from T_cam_obj on an RGB image."""
        vis = rgb.copy()

        # draw contour (yellow)
        if self.pose_vis_draw_contour and mask is not None:
            mask_u8 = (mask.astype(np.uint8) * 255)
            res = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = res[0] if len(res) == 2 else res[1]
            if contours:
                cv2.drawContours(vis, contours, -1, (255, 255, 0), 2)  # RGB yellow

        # axes
        R = T_cam_obj[:3, :3]
        t = T_cam_obj[:3, 3]
        L = float(self.pose_vis_axis_len)

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

        th = int(self.pose_vis_thickness)
        if p0 is not None:
            cv2.circle(vis, p0, max(2, th + 1), (255, 255, 255), -1)  # white

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
            #t = Time.from_msg(stamp)
            tf = self.tf_buffer.lookup_transform(self.base_frame, cam_frame, Time(), timeout=Duration(seconds=0.1))
            return tf_to_T(tf)
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed: {self.base_frame} <- {cam_frame} at stamp={stamp.sec}.{stamp.nanosec}: {e}"
            )
            return None

    def _on_synced(self, color_msg: Image, depth_msg: Image, mask_msg: Image):
        if self.K is None:
            return
        if self.mesh is None:
            self._set_status("READY")
            return

        with self.lock:
            if self.busy:
                return
            self.busy = True

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
                self._set_status("ERROR")
                return

            # choose target id
            obj_id = self.target_object_id
            if obj_id <= 0:
                obj_id = self._choose_target_id(inst)
            if obj_id == 0:
                self._set_status("NO_TARGET")
                self.initialized = False
                return

            mask = (inst == obj_id)
            if int(mask.sum()) < self.min_mask_pixels:
                self._set_status("NO_TARGET")
                self.initialized = False
                return
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
                        self._set_status("NO_TARGET")
                        return
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

            # --- TF: base <- camera ---
            T_base_cam = self._lookup_T_base_cam(cam_frame, color_msg.header.stamp)
            if T_base_cam is None:
                self._set_status("ERROR")
                return

            T_base_obj = T_base_cam @ T_cam_obj

            self._publish_pose(self.pub_T_base_obj, self.base_frame, color_msg.header.stamp, T_base_obj)
            self._publish_pose(self.pub_T_cam_obj, cam_frame, color_msg.header.stamp, T_cam_obj)

            # optional TF object_<id>
            if self.publish_object_tf:
                self._broadcast_object_tf(obj_id, self.base_frame, color_msg.header.stamp, T_base_obj)

            T_base_tcp = self._compute_target_tcp_T(T_base_obj)
            self._publish_pose(self.pub_target_tcp, self.base_frame, color_msg.header.stamp, T_base_tcp)

            if self.publish_pose_vis:
                vis = self._make_pose_vis(rgb, mask, K, T_cam_obj)
                self._publish_rgb8_image(self.pub_pose_vis, color_msg.header, vis)

            self._set_status("RUNNING")
        except Exception as e:
            self.get_logger().error(f"_on_synced exception: {e}\n{traceback.format_exc()}")
            self._set_status("ERROR")
            self.initialized = False
        finally:
            with self.lock:
                self.busy = False
    
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

        # 3) log 핵심 지표
        self.get_logger().info(
            f"z_med={z_med:.3f} valid_ratio={valid_ratio:.2f} z_med_valid={z_med_valid:.3f} T_cam_obj_z={T_cam_obj_z:.3f} "
            f"dist_px={dist_px:.1f}"
        )

        # 4) health check -> tracking reset trigget
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
            self.get_logger().warn(
                "diag: tracking unhealthy -> force re-register next frame "
                f"(valid_ratio={valid_ratio:.2f}, z_diff={(abs(z_med_valid - T_cam_obj_z) if np.isfinite(z_med_valid) else -1):.3f}, "
                f"dist_px={dist_px:.1f}"
            )

        return bad

def main():
    ros_args = sys.argv[:1] + ["--ros-args", "-r", "__ns:=/pose_tracker", "-r", "tf:=/tf", "-r", "tf_static:=/tf_static"]
    rclpy.init()
    node = PoseTrackerFoundationPose()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
