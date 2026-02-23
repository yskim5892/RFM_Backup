import argparse, json, math, os, random
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import numpy as np
import trimesh

# -----------------------------
# Utilities
# -----------------------------
def unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n

def orthonormal_frame_from_axes(closing_axis, approach_hint):
    """
    Return R (3x3) where:
      x = closing axis (finger closing direction, from contact1 to contact2)
      z = approach axis (gripper approach direction)
      y = z x x
    """
    x = unit(closing_axis)
    if x is None:
        return None

    # project approach_hint to be orthogonal to x
    ah = approach_hint - np.dot(approach_hint, x) * x
    z = unit(ah)
    if z is None:
        # fallback: pick any vector not parallel to x
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, x)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        z = unit(np.cross(x, tmp))
        if z is None:
            return None

    y = unit(np.cross(z, x))
    if y is None:
        return None
    # re-orthogonalize z
    z = unit(np.cross(x, y))
    if z is None:
        return None

    R = np.column_stack([x, y, z])
    return R

def make_T(R, t):
    T = np.eye(4, dtype=float)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def sample_in_cone(axis, cone_half_angle_rad):
    """
    Sample a random unit direction within a cone around 'axis' (unit vector).
    """
    axis = unit(axis)
    if axis is None:
        return None

    # sample angle from [0, cone_half_angle], uniform over cone solid angle
    u = random.random()
    cos_theta = 1 - u * (1 - math.cos(cone_half_angle_rad))
    sin_theta = math.sqrt(max(0.0, 1 - cos_theta*cos_theta))
    phi = 2 * math.pi * random.random()

    # build orthonormal basis (axis, b1, b2)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, axis)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    b1 = unit(np.cross(axis, tmp))
    b2 = unit(np.cross(axis, b1))

    d = cos_theta * axis + sin_theta * (math.cos(phi)*b1 + math.sin(phi)*b2)
    return unit(d)

def farthest_point_downsample(points, k):
    """Simple FPS on grasp centers to increase diversity."""
    if len(points) <= k:
        return list(range(len(points)))
    pts = np.asarray(points)
    idx = [random.randrange(len(pts))]
    d2 = np.sum((pts - pts[idx[0]])**2, axis=1)
    for _ in range(1, k):
        i = int(np.argmax(d2))
        idx.append(i)
        d2 = np.minimum(d2, np.sum((pts - pts[i])**2, axis=1))
    return idx


def project_contacts_to_surface(mesh: trimesh.Trimesh, candidates: list):
    """
    Snap contact points to nearest mesh surface for numerically stable output.
    Returns per-point projection distances (shape: 2 * len(candidates)).
    """
    if not candidates:
        return np.zeros((0,), dtype=float)

    points = np.zeros((2 * len(candidates), 3), dtype=float)
    for i, g in enumerate(candidates):
        points[2 * i] = g.c1
        points[2 * i + 1] = g.c2

    closest, dist, _ = trimesh.proximity.closest_point(mesh, points)
    for i, g in enumerate(candidates):
        g.c1 = closest[2 * i]
        g.c2 = closest[2 * i + 1]
        g.width = float(np.linalg.norm(g.c2 - g.c1))
        g.T_obj_tcp[:3, 3] = 0.5 * (g.c1 + g.c2)

    return dist


def load_grasps_json(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    if "grasps" not in data or not isinstance(data["grasps"], list):
        raise RuntimeError(f"Invalid grasp json (missing list 'grasps'): {json_path}")
    return data


def _load_mesh_clean(mesh_abs_path: str) -> trimesh.Trimesh:
    mesh_obj = trimesh.load(mesh_abs_path, force="mesh")
    if not isinstance(mesh_obj, trimesh.Trimesh):
        raise RuntimeError("Failed to load mesh as Trimesh")

    mesh_obj.update_faces(mesh_obj.unique_faces())
    mesh_obj.update_faces(mesh_obj.nondegenerate_faces())
    mesh_obj.remove_unreferenced_vertices()
    mesh_obj.process(validate=True)
    return mesh_obj


def load_topk_grasps(obj_name: str, topk: int):
    json_path = resolve_grasps_json_path(obj_name=obj_name)
    if not Path(json_path).exists():
        sample_and_save_grasps(obj_name=obj_name, topk=topk)

    data = load_grasps_json(json_path)
    grasps = data["grasps"]
    topk_saved = int(data.get("topk_saved", topk))
    if topk_saved <= 0:
        return grasps
    return grasps[:min(topk_saved, len(grasps))]


def resolve_grasps_json_path(obj_name: str | None = None, grasps_json_path: str | None = None, *, create_parent: bool = False):
    """
    Resolve grasps JSON path.
    - If grasps_json_path is given, return its absolute path.
    - Otherwise resolve to GraspPoseSampler/grasps/{obj_name}.json.
    """
    if grasps_json_path is not None:
        out_abs_path = os.path.abspath(grasps_json_path)
    else:
        obj_name = (obj_name or "").strip()
        if not obj_name:
            raise RuntimeError("Either `grasps_json_path` or `obj_name` must be provided.")
        obj_norm = _normalize_obj_name_for_path(obj_name)
        if not obj_norm:
            raise RuntimeError(f"Invalid obj_name='{obj_name}'")
        out_abs_path = os.path.abspath(str(Path(__file__).resolve().parent / "grasps" / f"{obj_norm}.json"))

    if create_parent:
        os.makedirs(os.path.dirname(out_abs_path), exist_ok=True)
    return out_abs_path

_PROJECT_UTILS_MODULE = None

def _get_project_utils_module():
    global _PROJECT_UTILS_MODULE
    if _PROJECT_UTILS_MODULE is not None:
        return _PROJECT_UTILS_MODULE

    repo_root = Path(__file__).resolve().parent.parent
    utils_path = repo_root / "utils.py"
    if not utils_path.exists():
        raise RuntimeError(f"Project utils.py not found: {utils_path}")

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    existing = sys.modules.get("utils")
    if existing is not None:
        existing_path = Path(getattr(existing, "__file__", "")).resolve()
        if existing_path == utils_path.resolve():
            _PROJECT_UTILS_MODULE = existing
            return existing
        del sys.modules["utils"]

    module = importlib.import_module("utils")
    module_path = Path(getattr(module, "__file__", "")).resolve()
    if module_path != utils_path.resolve():
        raise RuntimeError(f"Resolved wrong utils module: {module_path}")
    _PROJECT_UTILS_MODULE = module
    return module


def _normalize_obj_name_for_path(obj_name: str) -> str:
    text = (obj_name or "").strip()
    if not text:
        return ""
    try:
        project_utils = _get_project_utils_module()
        return project_utils.normalize_object_name(text)
    except Exception:
        return text.lower().replace(" ", "_")


def resolve_mesh_and_out_paths(obj_name: str | None = None, mesh: str | None = None, out: str | None = None):
    """
    Resolve mesh/out paths.
    - If mesh is missing, use utils.find_mesh_file(obj_name).
    - If out is missing, use GraspPoseSampler/grasps/{obj_name}.json.
    Returns: (mesh_input_path, mesh_abs_path, out_abs_path)
    """
    obj_name = (obj_name or "").strip() or None

    if mesh is None:
        if obj_name is None:
            raise RuntimeError("Either `mesh` or `obj_name` must be provided.")
        try:
            project_utils = _get_project_utils_module()
        except Exception as e:
            raise RuntimeError(
                f"Failed to import utils.find_mesh_file for obj_name='{obj_name}': {e}. "
                "Please provide `mesh` explicitly."
            ) from e
        mesh_path = project_utils.find_mesh_file(obj_name)
        if mesh_path is None:
            raise RuntimeError(f"Mesh not found for obj_name='{obj_name}' via utils.find_mesh_file")
        mesh = str(mesh_path)

    mesh_input_path = mesh
    mesh_abs_path = os.path.abspath(mesh)

    out_abs_path = resolve_grasps_json_path(
        obj_name=obj_name,
        grasps_json_path=out,
        create_parent=True,
    )
    return mesh_input_path, mesh_abs_path, out_abs_path


def sample_and_save_grasps(
    obj_name: str | None = None,
    mesh: str | None = None,
    out: str | None = None,
    *,
    num: int = 200,
    surface_pts: int = 2000,
    spp: int = 3,
    topk: int = 20,
    mu: float = 0.5,
    wmin: float = 0.02,
    wmax: float = 0.14,
    no_project_contacts: bool = False,
    seed: int = 0,
):
    """
    External entrypoint to sample grasps and save JSON.
    `mesh` and `out` can be omitted when `obj_name` is provided.
    """
    mesh_input_path, mesh_abs_path, out_abs_path = resolve_mesh_and_out_paths(
        obj_name=obj_name, mesh=mesh, out=out
    )

    # Keep raw mesh frame (no rezero) to match pose-tracker/FoundationPose frame.
    mesh_obj = _load_mesh_clean(mesh_abs_path)

    grasps = generate_antipodal_grasps(
        mesh_obj,
        num_targets=num,
        max_surface_points=surface_pts,
        samples_per_point=spp,
        mu=mu,
        width_min=wmin,
        width_max=wmax,
        seed=seed,
    )

    if not no_project_contacts and len(grasps) > 0:
        try:
            d = project_contacts_to_surface(mesh_obj, grasps)
            print(f"[INFO] projected contacts to surface (min/mean/max dist: {d.min():.6e} / {d.mean():.6e} / {d.max():.6e})")
        except Exception as e:
            print(f"[WARN] failed to project contacts to surface: {e}")

    if topk <= 0:
        grasps_to_save = grasps
    else:
        grasps_to_save = grasps[:min(topk, len(grasps))]

    out_data = {
        "obj_name": obj_name,
        "mesh": mesh_abs_path,
        "mesh_input": mesh_input_path,
        "mu": mu,
        "topk_saved": topk,
        "num_grasps_generated": len(grasps),
        "num_grasps": len(grasps_to_save),
        "frame": "object",
        "mesh_coord_frame": "raw_mesh",
        "T_obj_tcp_convention": {
            "x": "closing_axis (c1->c2)",
            "z": "approach_axis",
            "y": "z x x"
        },
        "grasps": []
    }

    for i, g in enumerate(grasps_to_save):
        out_data["grasps"].append({
            "id": i,
            "quality": g.quality,
            "width": g.width,
            "c1": g.c1.tolist(),
            "c2": g.c2.tolist(),
            "n1": g.n1.tolist(),
            "n2": g.n2.tolist(),
            "T_obj_tcp": g.T_obj_tcp.tolist()
        })

    with open(out_abs_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"[OK] wrote {out_abs_path} (saved={len(grasps_to_save)}, generated={len(grasps)})")
    return out_abs_path, out_data

# -----------------------------
# Main antipodal generator
# -----------------------------
@dataclass
class GraspCandidate:
    T_obj_tcp: np.ndarray  # 4x4 (object -> gripper tcp frame)
    quality: float
    c1: np.ndarray
    c2: np.ndarray
    n1: np.ndarray
    n2: np.ndarray
    width: float

def generate_antipodal_grasps(
    mesh: trimesh.Trimesh,
    num_targets: int = 200,
    max_surface_points: int = 2000,
    samples_per_point: int = 3,
    mu: float = 0.5,
    width_min: float = 0.02,
    width_max: float = 0.12,
    backoff: float = 0.002,
    max_ray_dist: float = 0.25,
    diversity_k: int = 200,
    seed: int = 0,
):
    """
    Dex-Net 스타일:
      - 표면 점 p1 샘플
      - friction cone 내 방향 d 샘플
      - p1에서 d 방향으로 ray cast -> p2 얻기
      - (p1,p2)에서의 법선이 서로 friction cone 조건을 만족하면 채택
    """
    random.seed(seed)
    np.random.seed(seed)

    # friction cone half-angle: arctan(mu)
    cone = math.atan(mu)

    # surface sampling (even-ish)
    # trimesh.sample.sample_surface_even은 version에 따라 다를 수 있어,
    # 없으면 sample_surface로 fallback합니다.
    if hasattr(trimesh.sample, "sample_surface_even"):
        pts, face_idx = trimesh.sample.sample_surface_even(mesh, max_surface_points)
    else:
        pts, face_idx = trimesh.sample.sample_surface(mesh, max_surface_points)

    face_normals = mesh.face_normals
    n1s = face_normals[face_idx]

    # ray intersector
    try:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    except Exception:
        intersector = mesh.ray  # slower fallback

    candidates: list[GraspCandidate] = []

    for p1, n1 in zip(pts, n1s):
        n1 = unit(n1)
        if n1 is None:
            continue

        # "closing direction" 후보: -n1 중심으로 friction cone에서 샘플
        for _ in range(samples_per_point):
            d = sample_in_cone(-n1, cone)
            if d is None:
                continue

            # backoff: 표면에서 살짝 띄워서 self-intersection 줄이기
            origin = p1 + backoff * d

            # intersect: origin + t*d
            locs, idx_ray, idx_tri = intersector.intersects_location(
                ray_origins=np.array([origin]),
                ray_directions=np.array([d]),
                multiple_hits=False
            )
            if len(locs) == 0:
                continue
            p2 = locs[0]
            dist = np.linalg.norm(p2 - p1)
            if not (width_min <= dist <= width_max) or dist > max_ray_dist:
                continue

            # get normal at p2 (use face normal)
            n2 = unit(mesh.face_normals[idx_tri[0]])
            if n2 is None:
                continue

            # antipodal check (friction cone)
            # Condition: closing axis aligns with inward direction at each contact within cone
            axis12 = unit(p2 - p1)  # from c1 to c2
            if axis12 is None:
                continue

            # At contact1, inward direction is +axis12, should be within cone around -n1
            if math.acos(np.clip(np.dot(axis12, -n1), -1, 1)) > cone:
                continue
            # At contact2, inward direction is -axis12, should be within cone around -n2
            if math.acos(np.clip(np.dot(-axis12, -n2), -1, 1)) > cone:
                continue

            # Build grasp pose (object -> gripper tcp)
            center = 0.5 * (p1 + p2)
            # approach hint: oppose average normal (towards object)
            approach_hint = -unit(n1 + n2) if unit(n1 + n2) is not None else -n1
            R = orthonormal_frame_from_axes(axis12, approach_hint)
            if R is None:
                continue

            T = make_T(R, center)

            # Simple quality: prefer more antipodal (smaller angles) and moderate width
            a1 = math.acos(np.clip(np.dot(axis12, -n1), -1, 1))
            a2 = math.acos(np.clip(np.dot(-axis12, -n2), -1, 1))
            q = float((cone - a1) + (cone - a2))  # bigger is better

            candidates.append(GraspCandidate(
                T_obj_tcp=T,
                quality=q,
                c1=p1, c2=p2,
                n1=n1, n2=n2,
                width=dist
            ))

    # sort by quality
    candidates.sort(key=lambda g: g.quality, reverse=True)

    # diversify using FPS on centers
    centers = [g.T_obj_tcp[:3, 3] for g in candidates]
    keep_idx = farthest_point_downsample(centers, min(diversity_k, len(candidates)))
    candidates = [candidates[i] for i in keep_idx]
    candidates.sort(key=lambda g: g.quality, reverse=True)

    return candidates[:num_targets]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj_name", default=None, help="Object name (used to auto-resolve mesh/out)")
    ap.add_argument("--mesh", default=None, help="Path to mesh (obj/ply/stl...). Optional if --obj_name is set")
    ap.add_argument("--out", default=None, help="Output JSON. Optional if --obj_name is set")
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--surface_pts", type=int, default=2000)
    ap.add_argument("--spp", type=int, default=3, help="samples per surface point")
    ap.add_argument("--topk", type=int, default=20,
                    help="Save only top-k grasps to JSON (<=0 means save all)")
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--wmin", type=float, default=0.02)
    ap.add_argument("--wmax", type=float, default=0.14)
    ap.add_argument("--no_project_contacts", action="store_true",
                    help="Disable projection of c1/c2 to nearest mesh surface before saving")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sample_and_save_grasps(
        obj_name=args.obj_name,
        mesh=args.mesh,
        out=args.out,
        num=args.num,
        surface_pts=args.surface_pts,
        spp=args.spp,
        topk=args.topk,
        mu=args.mu,
        wmin=args.wmin,
        wmax=args.wmax,
        no_project_contacts=args.no_project_contacts,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
