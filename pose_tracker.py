#!/usr/bin/env python3
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import utils


class PoseTracker:
    def __init__(self, repo_root: Path, logger):
        self.repo_root = Path(repo_root)
        self.foundationpose_root = self.repo_root / "thirdparty" / "FoundationPose"
        self.logger = logger

        self._backend_loaded = False
        self.torch = None
        self.trimesh = None
        self.FoundationPose = None
        self.scorer = None
        self.refiner = None
        self.glctx = None
        self.fp_debug_dir = "/tmp/foundationpose_debug"

        self.foundation_pose = None
        self.initialized = False
        self.est_refine_iter = 5
        self.track_refine_iter = 1

        self.T_obb_mesh: Optional[np.ndarray] = None
        self.mesh_centroid_obj: Optional[np.ndarray] = None

    @staticmethod
    def normalize_object_name(name: str) -> str:
        return (name or "").strip().lower().replace(" ", "_")

    def _find_mesh_file(self, object_name: str) -> Optional[Path]:
        norm = self.normalize_object_name(object_name)
        if not norm:
            return None

        ycb_root = self.repo_root / "ycb"
        if not ycb_root.exists():
            return None

        matches = sorted(p for p in ycb_root.glob(f"*{norm}*") if p.is_dir())
        if not matches:
            return None

        mesh_file = matches[0] / "google_16k" / "textured.obj"
        if not mesh_file.exists():
            return None
        return mesh_file

    def mesh_exists(self, object_name: str) -> bool:
        return self._find_mesh_file(object_name) is not None

    def _ensure_backend_loaded(self):
        if self._backend_loaded:
            return

        if str(self.foundationpose_root) not in sys.path:
            sys.path.insert(0, str(self.foundationpose_root))

        import torch
        import trimesh
        import nvdiffrast.torch as dr
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

        logging.getLogger().setLevel(logging.ERROR)

        self.torch = torch
        self.trimesh = trimesh
        self.FoundationPose = FoundationPose
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self._backend_loaded = True

    def set_object(self, object_name: str) -> bool:
        mesh_file = self._find_mesh_file(object_name)
        if mesh_file is None:
            return False

        self._ensure_backend_loaded()

        mesh = self.trimesh.load(str(mesh_file))
        mesh.vertices = mesh.vertices.astype(np.float64)
        if getattr(mesh, "vertex_normals", None) is not None:
            mesh.vertex_normals = mesh.vertex_normals.astype(np.float64)

        model_normals = mesh.vertex_normals if getattr(mesh, "vertex_normals", None) is not None else np.zeros_like(mesh.vertices)

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

        try:
            self.T_obb_mesh, _ = self.trimesh.bounds.oriented_bounds(mesh)
        except Exception:
            self.T_obb_mesh = np.eye(4)

        min_bound = mesh.vertices.min(axis=0)
        max_bound = mesh.vertices.max(axis=0)
        self.mesh_centroid_obj = (min_bound + max_bound) / 2.0

        self.initialized = False

        self.logger.info(f"Loaded '{object_name}': centroid={self.mesh_centroid_obj}")
        return True

    def clear_object(self):
        self.initialized = False
        self.T_obb_mesh = None
        self.mesh_centroid_obj = None

    def reset_tracking(self):
        self.initialized = False

    def track_pose(self, K, rgb, depth, mask):
        if self.foundation_pose is None:
            return None

        with self.torch.inference_mode():
            if not self.initialized:
                T_cam_obj = self.foundation_pose.register(K, rgb, depth, mask, self.est_refine_iter)
                if T_cam_obj is None:
                    return None
                self.initialized = True
            else:
                T_cam_obj = self.foundation_pose.track_one(rgb, depth, K, self.track_refine_iter)

        T_cam_obj = np.asarray(T_cam_obj, dtype=np.float64)
        if T_cam_obj.shape != (4, 4):
            return None
        return T_cam_obj

    def overlay_pose_axes_and_center(
        self,
        base_vis: np.ndarray,
        K: np.ndarray,
        T_cam_obj: np.ndarray,
        x_axis_cam: Optional[np.ndarray],
    ) -> np.ndarray:
        vis = base_vis.copy()

        R = T_cam_obj[:3, :3]
        t = T_cam_obj[:3, 3]
        L = 0.05

        pts_obj = np.array(
            [
                [0.0, 0.0, 0.0],
                [L, 0.0, 0.0],
                [0.0, L, 0.0],
                [0.0, 0.0, L],
            ],
            dtype=np.float64,
        )

        pts_cam = (R @ pts_obj.T).T + t.reshape(1, 3)

        p0 = utils.project_uv(K, pts_cam[0])
        px = utils.project_uv(K, pts_cam[1])
        py = utils.project_uv(K, pts_cam[2])
        pz = utils.project_uv(K, pts_cam[3])

        th = 3
        if p0 is not None:
            cv2.circle(vis, p0, max(2, th + 1), (255, 255, 255), -1)

        p_center_cam = None
        if self.mesh_centroid_obj is not None:
            p_center_cam = (T_cam_obj[:3, :3] @ self.mesh_centroid_obj) + T_cam_obj[:3, 3]
            p_center_uv = utils.project_uv(K, p_center_cam)
            if p_center_uv is not None:
                cv2.circle(vis, p_center_uv, 5, (0, 255, 255), -1)

        if x_axis_cam is not None:
            start_cam = p_center_cam if p_center_cam is not None else t
            end_cam = start_cam + x_axis_cam * 0.08

            uv_start = utils.project_uv(K, start_cam)
            uv_end = utils.project_uv(K, end_cam)
            if uv_start is not None and uv_end is not None:
                cv2.line(vis, uv_start, uv_end, (255, 0, 255), 2)

        if p0 is not None and px is not None:
            cv2.line(vis, p0, px, (255, 0, 0), th)
        if p0 is not None and py is not None:
            cv2.line(vis, p0, py, (0, 255, 0), th)
        if p0 is not None and pz is not None:
            cv2.line(vis, p0, pz, (0, 0, 255), th)
        return vis

    # FoundationPose가 계산한 6d pos값과, mask centroid/depth 값이 차이가 너무 많이 나면 bad return
    def check_pose_valid(self, T_cam_obj, K, centroid_uv, z_mean):
        T_cam_obj_z = float(T_cam_obj[2, 3])

        bad = False
        if not np.isfinite(z_mean):
            bad = True
        if np.isfinite(z_mean) and abs(z_mean - T_cam_obj_z) > 0.1:
            bad = True

        uv_pose = utils.project_uv(K, T_cam_obj[:3, 3].astype(np.float64, copy=False))
        if uv_pose is not None and centroid_uv is not None:
            u_pose, v_pose = uv_pose
            u_mask, v_mask = centroid_uv
            dist_px = float(np.hypot(u_pose - float(u_mask), v_pose - float(v_mask)))
        else:
            dist_px = float("nan")
        if np.isfinite(dist_px) and dist_px > 120.0:
            bad = True

        return bad

    @staticmethod
    def compute_horizontal_x_axis(
        mask: np.ndarray,
        K: np.ndarray,
        T_base_cam: np.ndarray,
        z_ref_m: float,
        latest_tcp_R_base: np.ndarray,
    ) -> np.ndarray:
        def _fallback_x_target() -> np.ndarray:
            x = latest_tcp_R_base[:, 0].astype(np.float64, copy=True)
            x[2] = 0.0
            n = np.linalg.norm(x)
            if n < 1e-9:
                return np.array([1.0, 0.0, 0.0], dtype=np.float64)
            return x / n

        if mask is None:
            return _fallback_x_target()

        mask_u8 = mask.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask_er = cv2.erode(mask_u8, kernel, iterations=1).astype(bool)
        ys, xs = np.nonzero(mask_er)
        if xs.size < 10:
            return _fallback_x_target()

        pts = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
        pts_centered = pts - pts.mean(axis=0, keepdims=True)
        cov = (pts_centered.T @ pts_centered) / max(1, pts_centered.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)

        idx_short = int(np.argmin(eigvals))
        v_short_img = eigvecs[:, idx_short]

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        z_use = float(z_ref_m)
        if (not np.isfinite(z_use)) or z_use <= 1e-6 or fx <= 1e-6 or fy <= 1e-6:
            x_target = _fallback_x_target()
        else:
            v_short_cam = np.array(
                [v_short_img[0] * z_use / fx, v_short_img[1] * z_use / fy, 0.0],
                dtype=np.float64,
            )
            n_short = np.linalg.norm(v_short_cam)
            if n_short < 1e-9:
                x_target = _fallback_x_target()
            else:
                v_short_cam /= n_short
                R_base_cam = T_base_cam[:3, :3]
                x_target = R_base_cam @ v_short_cam

        x_target = np.asarray(x_target, dtype=np.float64).copy()
        x_target[2] = 0.0
        n_target = np.linalg.norm(x_target)
        if n_target < 1e-9:
            x_target = _fallback_x_target()
        else:
            x_target /= n_target

        current_x = latest_tcp_R_base[:, 0].astype(np.float64, copy=True)
        if np.dot(x_target, current_x) < 0:
            x_target = -x_target

        return x_target

    def compute_target_tcp_T(
        self,
        T_base_obj: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        T_base_cam: np.ndarray,
        z_ref_m: float,
        latest_tcp_R_base: np.ndarray,
        latest_tcp_p_base: np.ndarray,
        pregrasp_height: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.mesh_centroid_obj is not None:
            p_obj_base = (T_base_obj[:3, :3] @ self.mesh_centroid_obj) + T_base_obj[:3, 3]
        else:
            p_obj_base = T_base_obj[:3, 3]

        if self.T_obb_mesh is None:
            R_tcp = latest_tcp_R_base
            z_tcp_base = R_tcp[:, 2]
            z_tcp_base /= (np.linalg.norm(z_tcp_base) + 1e-12)
            p_tcp = p_obj_base - z_tcp_base * pregrasp_height
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_tcp
            T[:3, 3] = p_tcp
            x_axis_base = self.compute_horizontal_x_axis(mask, K, T_base_cam, z_ref_m, latest_tcp_R_base)
            return T, x_axis_base

        x_axis_base = self.compute_horizontal_x_axis(mask, K, T_base_cam, z_ref_m, latest_tcp_R_base)

        p_current = latest_tcp_p_base
        v_approach = p_obj_base - p_current
        dist = np.linalg.norm(v_approach)
        if dist > 1e-3:
            v_approach /= dist
        else:
            v_approach = np.array([0.0, 0.0, -1.0])

        dot_val = np.dot(v_approach, x_axis_base)
        z_raw = v_approach - dot_val * x_axis_base

        if np.linalg.norm(z_raw) < 0.1:
            z_target = np.array([0.0, 0.0, -1.0])
        else:
            z_target = z_raw / np.linalg.norm(z_raw)

        y_target = np.cross(z_target, x_axis_base)
        R_tcp = np.column_stack((x_axis_base, y_target, z_target))

        z_wrist_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        z_wrist_in_base = R_tcp @ z_wrist_local
        p_tcp = p_obj_base - z_wrist_in_base * pregrasp_height

        if p_tcp[2] < 0.2:
            p_tcp[2] = 0.2

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_tcp
        T[:3, 3] = p_tcp
        return T, x_axis_base
