#!/usr/bin/env python3
import os
import sys
import threading
from typing import Optional, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import String, Int32
from std_srvs.srv import Trigger

from cv_bridge import CvBridge
import message_filters

from math_utils import quat_to_mat, mat_to_quat, tf_to_T, pose_to_T

import tf2_ros


# ----------------- PoseTracker -----------------
class PoseTrackerFoundationPose(Node):
    def __init__(self):
        super().__init__("pose_tracker_foundationpose")

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.busy = False

        # -------- params --------
        self.foundationpose_root = self.declare_parameter(
            "foundationpose_root", os.environ.get("FOUNDATIONPOSE_ROOT", "/workspace/FoundationPose")
        ).value
        self.mesh_file = self.declare_parameter("mesh_file", "").value  # required
        self.base_frame = self.declare_parameter("base_frame", "base_link").value
        self.camera_frame = self.declare_parameter("camera_frame", "").value  # empty -> use msg.header.frame_id

        self.depth_scale = float(self.declare_parameter("depth_scale", 0.001).value)  # uint16(mm)->m

        self.target_object_id = int(self.declare_parameter("target_object_id", 0).value)  # 0 => auto(largest)
        self.min_mask_pixels = int(self.declare_parameter("min_mask_pixels", 200).value)

        self.est_refine_iter = int(self.declare_parameter("est_refine_iter", 5).value)
        self.track_refine_iter = int(self.declare_parameter("track_refine_iter", 2).value)

        # TCP heuristic
        self.pregrasp_height = float(self.declare_parameter("pregrasp_height", 0.10).value)  # meters above object

        self.publish_object_tf = bool(self.declare_parameter("publish_object_tf", True).value)

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

        if not self.mesh_file:
            raise RuntimeError('mesh_file param is required (e.g. /path/to/textured.obj)')

        # -------- TF --------
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # -------- outputs (relative to /pose_tracker namespace) --------
        self.pub_object_pose = self.create_publisher(PoseStamped, "object_pose", 10)
        self.pub_target_tcp = self.create_publisher(PoseStamped, "target_tcp_pose", 10)
        self.pub_status = self.create_publisher(String, "status", 10)

        # optional input: /pose_tracker/target_object_id
        self.sub_target_id = self.create_subscription(Int32, "target_object_id", self._on_target_id, 10)

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
        self.dr = dr
        self.FoundationPose = FoundationPose
        self.ScorePredictor = ScorePredictor
        self.PoseRefinePredictor = PoseRefinePredictor

        self.get_logger().info(f"Loading mesh: {self.mesh_file}")
        self.mesh = self.trimesh.load(self.mesh_file)

        self.scorer = self.ScorePredictor()
        self.refiner = self.PoseRefinePredictor()
        self.glctx = self.dr.RasterizeCudaContext()

        self.est = self.FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir="/tmp/foundationpose_debug",
            debug=0,
            glctx=self.glctx,
        )

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
            f"  Pub /pose_tracker/object_pose, /pose_tracker/target_tcp_pose, /pose_tracker/status\n"
        )

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
        # FP 내부 reset(있으면 호출)
        if hasattr(self.est, "reset"):
            try:
                self.est.reset()
            except Exception:
                pass
        resp.success = True
        resp.message = "reset ok"
        return resp

    def _on_cam_info(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)

    def _decode_rgb(self, msg: Image) -> np.ndarray:
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            return rgb
        except Exception:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _decode_depth_m(self, msg: Image) -> np.ndarray:
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if d.ndim == 3:
            d = d[:, :, 0]
        if d.dtype == np.uint16:
            depth_m = d.astype(np.float32) * self.depth_scale
        else:
            depth_m = d.astype(np.float32)
        depth_m[~np.isfinite(depth_m)] = 0.0
        return depth_m

    def _decode_instance_mask(self, msg: Image) -> np.ndarray:
        m = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if m.ndim == 3:
            m = m[:, :, 0]
        if m.dtype != np.int32:
            m = m.astype(np.int32)
        return m

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
            return tf_to_T(tf)
        except Exception:
            return None

    def _on_synced(self, color_msg: Image, depth_msg: Image, mask_msg: Image):
        if self.K is None:
            return

        with self.lock:
            if self.busy:
                return
            self.busy = True

        try:
            rgb = self._decode_rgb(color_msg)
            depth_m = self._decode_depth_m(depth_msg)
            inst = self._decode_instance_mask(mask_msg)

            # size safety
            if depth_m.shape[:2] != rgb.shape[:2]:
                depth_m = cv2.resize(depth_m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
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

            # --- FoundationPose ---
            with self.torch.inference_mode():
                if not self.initialized:
                    # register once using mask
                    try:
                        T_cam_obj = self.est.register(rgb=rgb, depth=depth_m, K=self.K,
                                                      mask=mask, iteration=self.est_refine_iter)
                    except TypeError:
                        T_cam_obj = self.est.register(rgb=rgb, depth=depth_m, K=self.K,
                                                      ob_mask=mask, iteration=self.est_refine_iter)

                    if T_cam_obj is None:
                        self._set_status("NO_TARGET")
                        return
                    self.initialized = True
                else:
                    T_cam_obj = self.est.track_one(rgb=rgb, depth=depth_m, K=self.K,
                                                   iteration=self.track_refine_iter)

            T_cam_obj = np.asarray(T_cam_obj, dtype=np.float64)
            if T_cam_obj.shape != (4, 4):
                self._set_status("ERROR")
                self.initialized = False
                return

            # --- TF: base <- camera ---
            T_base_cam = self._lookup_T_base_cam(cam_frame, color_msg.header.stamp)
            if T_base_cam is None:
                self._set_status("ERROR")
                return

            T_base_obj = T_base_cam @ T_cam_obj

            # publish object pose (frame_id=base_link 권장)
            self._publish_pose(self.pub_object_pose, self.base_frame, color_msg.header.stamp, T_base_obj)

            # optional TF object_<id>
            if self.publish_object_tf:
                self._broadcast_object_tf(obj_id, self.base_frame, color_msg.header.stamp, T_base_obj)

            # compute + publish target tcp pose (frame_id=base_link 필수)
            T_base_tcp = self._compute_target_tcp_T(T_base_obj)
            self._publish_pose(self.pub_target_tcp, self.base_frame, color_msg.header.stamp, T_base_tcp)

            self._set_status("RUNNING")

        except Exception:
            self._set_status("ERROR")
            self.initialized = False
        finally:
            with self.lock:
                self.busy = False


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

