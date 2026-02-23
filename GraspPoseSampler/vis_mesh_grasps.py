import argparse
import json
import os
import numpy as np
import trimesh

try:
    from grasp_sampler import resolve_grasps_json_path
except Exception:
    from GraspPoseSampler.grasp_sampler import resolve_grasps_json_path


def load_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Failed to load mesh as Trimesh: {mesh_path}")
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)
    return mesh


def load_grasps_json(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    grasps = data.get("grasps", [])
    if not grasps:
        raise RuntimeError(f"No grasps found in: {json_path}")
    return data, grasps


def resolve_mesh_path(data: dict, grasps_json_path: str, mesh_arg: str | None):
    """
    Resolve mesh path from CLI arg or grasp JSON metadata.
    """
    candidates = []
    if mesh_arg:
        candidates.append(mesh_arg)
    for key in ("mesh", "mesh_abs", "mesh_path", "obj_path", "mesh_input"):
        value = data.get(key, None)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    json_dir = os.path.dirname(os.path.abspath(grasps_json_path))
    tried = []
    for c in candidates:
        if os.path.isabs(c):
            tried.append(c)
            if os.path.exists(c):
                return c
        else:
            tried.append(c)
            if os.path.exists(c):
                return os.path.abspath(c)
            rel_to_json = os.path.abspath(os.path.join(json_dir, c))
            tried.append(rel_to_json)
            if os.path.exists(rel_to_json):
                return rel_to_json

    raise RuntimeError(
        "Failed to resolve mesh path. Provide --mesh explicitly or include a valid "
        f"mesh path in JSON metadata (checked: {tried})"
    )


def make_colored_segment_mesh(p0: np.ndarray, p1: np.ndarray, radius=0.001, rgba=(255, 255, 255, 255)):
    """
    Create a colored cylindrical segment. Unlike Path3D lines, this respects depth
    consistently in most viewers.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    if np.linalg.norm(p1 - p0) < 1e-12:
        return None
    seg = trimesh.creation.cylinder(radius=float(radius), segment=np.array([p0, p1]))
    seg.visual.vertex_colors = np.tile(np.array(rgba, dtype=np.uint8), (len(seg.vertices), 1))
    return seg


def make_axis_frame(T: np.ndarray, axis_len=0.05, radius=0.001):
    """
    Create a triad (x,y,z) at transform T.
    T: 4x4 object->frame transform (we place the triad in object frame).
    """
    origin = T[:3, 3]
    R = T[:3, :3]
    x_end = origin + axis_len * R[:, 0]
    y_end = origin + axis_len * R[:, 1]
    z_end = origin + axis_len * R[:, 2]

    geoms = []
    gx = make_colored_segment_mesh(origin, x_end, radius=radius, rgba=(255, 0, 0, 255))
    gy = make_colored_segment_mesh(origin, y_end, radius=radius, rgba=(0, 255, 0, 255))
    gz = make_colored_segment_mesh(origin, z_end, radius=radius, rgba=(0, 0, 255, 255))
    if gx is not None:
        geoms.append(gx)
    if gy is not None:
        geoms.append(gy)
    if gz is not None:
        geoms.append(gz)
    return geoms

def make_points(points: np.ndarray, radius=0.003, rgba=(255, 255, 0, 255)):
    """
    Visualize points as small icospheres combined into one mesh.
    """
    spheres = []
    for p in points:
        s = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        s.apply_translation(p)
        s.visual.vertex_colors = np.array(rgba, dtype=np.uint8)
        spheres.append(s)
    if not spheres:
        return None
    return trimesh.util.concatenate(spheres)

def parse_T(grasp_rec):
    T = np.array(grasp_rec["T_obj_tcp"], dtype=float)
    if T.shape != (4, 4):
        raise ValueError("T_obj_tcp must be 4x4")
    return T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj_name", default=None,
                    help="Object name used to auto-resolve grasps JSON path (GraspPoseSampler/grasps/{obj_name}.json)")
    ap.add_argument("--mesh", default=None, help="Path to object mesh (.obj/.ply/...). If omitted, use path stored in grasps JSON.")
    ap.add_argument("--grasps", default=None, help="Path to object_grasps.json. Optional if --obj_name is set")
    ap.add_argument("--topk", type=int, default=20, help="Show top-k grasps (sorted by quality in json)")
    ap.add_argument("--index", type=int, default=-1, help="Show only one grasp by index (overrides topk if >=0)")
    ap.add_argument("--axis_len", type=float, default=0.05, help="Axis length for grasp frames (meters)")
    ap.add_argument("--axis_radius", type=float, default=0.0007, help="Axis cylinder radius (meters)")
    ap.add_argument("--show_frames", action="store_true", help="Show grasp frame axes (x,y,z)")
    ap.add_argument("--show_origin", action="store_true", help="Show world origin axis")
    ap.add_argument("--show_contacts", action="store_true", help="Show contact segment c1-c2 if present")
    ap.add_argument("--contact_radius", type=float, default=0.0009, help="Radius for contact line/point rendering (meters)")
    ap.add_argument("--mesh_alpha", type=int, default=127,
                    help="Mesh transparency 0-255 (lower is more transparent; 255 is opaque)")
    args = ap.parse_args()

    # Default view focuses on contact geometry to avoid confusion with frame axes.
    if not args.show_frames and not args.show_contacts:
        args.show_contacts = True

    grasps_path = resolve_grasps_json_path(obj_name=args.obj_name, grasps_json_path=args.grasps)
    if not os.path.exists(grasps_path):
        raise RuntimeError(
            f"Grasps JSON not found: {grasps_path}. "
            "Provide --grasps explicitly or generate it first with sample_grasps_from_mesh.py."
        )

    data, grasps = load_grasps_json(grasps_path)
    mesh_path = resolve_mesh_path(data, grasps_path, args.mesh)
    mesh = load_mesh(mesh_path)

    # Scene
    scene = trimesh.Scene()

    # Make mesh semi-transparent for easier viewing
    # (Some viewers ignore alpha; still fine.)
    if mesh.visual.kind != "vertex":
        mesh.visual = mesh.visual.to_color()
    colors = mesh.visual.vertex_colors
    if colors is None or len(colors) == 0:
        mesh.visual.vertex_colors = np.tile(np.array([200, 200, 200, args.mesh_alpha], dtype=np.uint8),
                                            (len(mesh.vertices), 1))
    else:
        mesh.visual.vertex_colors[:, 3] = np.uint8(np.clip(args.mesh_alpha, 0, 255))
    scene.add_geometry(mesh, node_name="object_mesh")

    # Select grasps to show
    if args.index >= 0:
        if args.index >= len(grasps):
            raise RuntimeError(f"--index {args.index} out of range (0..{len(grasps)-1})")
        show = [(args.index, grasps[args.index])]
    else:
        k = min(args.topk, len(grasps))
        show = list(enumerate(grasps[:k]))

    contact_surface_dists = []

    # Add grasp visuals
    for idx, g in show:
        T = parse_T(g)
        # frame
        if args.show_frames:
            for axis_i, axis_geom in enumerate(make_axis_frame(T, axis_len=args.axis_len, radius=args.axis_radius)):
                scene.add_geometry(axis_geom, node_name=f"grasp_{idx}_frame_{axis_i}")

        # contacts
        has_c1 = "c1" in g and isinstance(g["c1"], list)
        has_c2 = "c2" in g and isinstance(g["c2"], list)
        if has_c1 and has_c2:
            c1 = np.array(g["c1"], dtype=float)
            c2 = np.array(g["c2"], dtype=float)
            pts = np.stack([c1, c2], axis=0)
            c1, c2 = pts[0], pts[1]

            if args.show_contacts:
                seg = make_colored_segment_mesh(c1, c2, radius=args.contact_radius, rgba=(255, 255, 0, 255))
                if seg is not None:
                    scene.add_geometry(seg, node_name=f"grasp_{idx}_contacts")

            p1 = make_points(np.array([c1]), radius=args.contact_radius, rgba=(255, 80, 80, 255))
            p2 = make_points(np.array([c2]), radius=args.contact_radius, rgba=(80, 200, 255, 255))
            if p1 is not None:
                scene.add_geometry(p1, node_name=f"grasp_{idx}_c1")
            if p2 is not None:
                scene.add_geometry(p2, node_name=f"grasp_{idx}_c2")

    if args.show_origin:
        origin = trimesh.creation.axis(origin_size=args.axis_len * 0.2, axis_length=args.axis_len * 1.2)
        scene.add_geometry(origin, node_name="world_origin")

    # Print a quick summary
    print(f"[INFO] mesh: {mesh_path}")
    print(f"[INFO] grasps json: {grasps_path}")
    print(f"[INFO] total grasps in json: {len(grasps)}")
    if args.index >= 0:
        q = grasps[args.index].get("quality", None)
        print(f"[INFO] showing grasp index={args.index} quality={q}")
    else:
        qs = [g.get("quality", None) for _, g in show]
        print(f"[INFO] showing topk={len(show)} (qualities head={qs[:5]})")
    print(f"[INFO] show_frames={args.show_frames}, show_contacts={args.show_contacts}, show_origin={args.show_origin}")
    if contact_surface_dists:
        d = np.array(contact_surface_dists, dtype=float)
        print(f"[INFO] contact->surface distance stats (min/mean/max): {np.nanmin(d):.6e} / {np.nanmean(d):.6e} / {np.nanmax(d):.6e}")

    scene.show()


if __name__ == "__main__":
    main()
