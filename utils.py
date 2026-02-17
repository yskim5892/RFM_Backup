import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped
from math_utils import mat_to_quat, quat_to_mat, tf_to_T
from sensor_msgs.msg import Image
from typing import Optional, Tuple

def label_to_rgb(label: np.ndarray) -> np.ndarray:
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
    ]
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(1, int(label.max()) + 1):
        rgb[label == i] = palette[(i - 1) % len(palette)]
    return rgb

def make_pose_msg(frame_id: str, stamp, T: np.ndarray) -> PoseStamped:
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


def decode_rgb(msg: Image) -> np.ndarray:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    enc = msg.encoding.lower()

    if enc in ("rgb8", "rgba8"):
        return img[:, :, :3]
    if enc in ("bgr8", "bgra8"):
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    raise RuntimeError(f"Unsupported color encoding: {msg.encoding}")


def decode_depth_m(msg: Image, depth_scale: float) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc in ("16uc1", "mono16"):
        d = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        return d.astype(np.float32) * depth_scale
    if enc in ("32fc1",):
        d = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        d[~np.isfinite(d)] = 0.0
        return d
    raise RuntimeError(f"Unsupported depth encoding: {msg.encoding}")


def decode_instance_mask(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc in ("32sc1",):
        return np.frombuffer(msg.data, dtype=np.int32).reshape(msg.height, msg.width)
    if enc in ("mono8", "8uc1"):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width).astype(np.int32)
    raise RuntimeError(f"Unsupported mask encoding: {msg.encoding}")


def make_imgmsg_32sc1(label: np.ndarray, header) -> Image:
    if label.dtype != np.int32:
        label = label.astype(np.int32)
    h, w = label.shape[:2]
    msg = Image()
    msg.header.stamp = header.stamp
    msg.header.frame_id = header.frame_id
    msg.height = int(h)
    msg.width = int(w)
    msg.encoding = "32SC1"
    msg.is_bigendian = False
    msg.step = int(w * 4)
    msg.data = label.tobytes()
    return msg


def make_imgmsg_rgb8(rgb: np.ndarray, header) -> Image:
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8, copy=False)
    h, w = rgb.shape[:2]
    msg = Image()
    msg.header.stamp = header.stamp
    msg.header.frame_id = header.frame_id
    msg.height = int(h)
    msg.width = int(w)
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = int(w * 3)
    msg.data = rgb.tobytes()
    return msg


def parse_prompt_to_object(prompt: str) -> str:
    text = (prompt or "").strip()
    if not text:
        return ""
    parts = text.split()
    if len(parts) <= 1:
        return ""
    return " ".join(parts[1:]).strip()

def parse_objects_to_track(raw_text: str) -> list[str]:
    text = (raw_text or "").strip()
    if not text:
        return []

    # 1) JSON list
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    # 2) Python literal list
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    # 3) comma or whitespace separated
    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]
    if " " in text:
        return [x.strip() for x in text.split() if x.strip()]

    # 4) single object string
    return [text]


def normalize_object_name(text: str) -> str:
    return (text or "").strip().lower().replace(" ", "_")


def publish_rgb8_image(pub, header, rgb: np.ndarray):
    if rgb is None:
        return
    pub.publish(make_imgmsg_rgb8(rgb, header))


def project_uv(K: np.ndarray, p_cam: np.ndarray) -> Optional[Tuple[int, int]]:
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
