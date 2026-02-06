#!/usr/bin/env python3
import os
from pathlib import Path
import threading
import json
import ast

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.ops import box_convert
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String as RosString
from std_msgs.msg import Header as RosHeader
from cv_bridge import CvBridge

# GSAM = GroundingDINO + SAM
from groundingdino.util.inference import load_model, predict
from groundingdino.datasets import transforms as T
from segment_anything import sam_model_registry, SamPredictor

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

from utils import label_to_rgb


def load_gsam(gd_config: str, gd_ckpt: str, sam_ckpt: str, device: str = "cuda"):
    dev = torch.device(device)
    gd_model = load_model(gd_config, gd_ckpt).to(dev).eval()

    gd_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to(dev).eval()
    sam_predictor = SamPredictor(sam)
    return gd_model, gd_transform, sam_predictor, dev

@torch.inference_mode()
def gsam_make_mask(
    image_bgr: np.ndarray,
    prompt: str,
    gd_model,
    gd_transform,
    sam_predictor,
    device: torch.device,
    topk: int = 1,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> np.ndarray:

    caption = prompt.strip()
    if not caption.endswith("."):
        caption += "."

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    pil = PILImage.fromarray(image_rgb)
    gd_img, _ = gd_transform(pil, None)  # (3,H,W) torch
    gd_img = gd_img.to(device)

    boxes, logits, _ = predict(
        model=gd_model,
        image=gd_img,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    idx = torch.argsort(logits, descending=True)[:topk]
    boxes = boxes[idx] 

    if boxes is None or len(boxes) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # GroundingDINO boxes: normalized cxcywh -> pixel xyxy
    scale = torch.tensor([w, h, w, h], device=boxes.device, dtype=boxes.dtype)
    boxes_xyxy = box_convert(boxes * scale, in_fmt="cxcywh", out_fmt="xyxy")
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0, w - 1)
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0, h - 1)

    sam_predictor.set_image(image_rgb)

    mask = np.zeros((h, w), dtype=np.int32)
    for i, b in enumerate(boxes_xyxy, start=1):
        m, _, _ = sam_predictor.predict(box=b.detach().cpu().numpy(), multimask_output=False)
        mask[m[0].astype(bool)] = i

    return mask


# -------------------------
# ROS2 노드: 첫 프레임 GSAM init → Cutie tracking → 15Hz publish
# -------------------------
class GSAMCutieTracker(Node):
    def __init__(self, args):
        super().__init__("gsam_cutie_tracker")
        self.args = args
        self.bridge = CvBridge()

        here = Path(__file__).resolve().parent / "thirdparty" / "Grounded-Segment-Anything"

        self.declare_parameter("publish_rate", 15.0)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("box_threshold", 0.35)
        self.declare_parameter("text_threshold", 0.25)
        self.declare_parameter("max_internal_size", 480)
        
        self.image_topic = "/wrist_cam/camera/color/image_raw"
        self.mask_topic = "/perception/instance_mask"
        self.init_mask_topic = "/perception/initial_mask"
        self.mask_vis_topic = "/perception/instance_mask_vis"
        self.init_mask_vis_topic = "/perception/initial_mask_vis"
        self.objects_to_track_topic = "/inference/objects_to_track"
        self._img_lock = threading.Lock()

        device = self.get_parameter("device").value
        if device.startswith("cuda") and not torch.cuda.is_available():
            self.get_logger().warning("CUDA not available -> fallback to cpu")
            device = "cpu"

        self.box_th = float(self.get_parameter("box_threshold").value)
        self.text_th = float(self.get_parameter("text_threshold").value)

        # subscribe/publish
        self.sub = self.create_subscription(
            RosImage, self.image_topic, self._on_image, qos_profile_sensor_data
        )
        # Object Mask를 Node끼리 주고받을 때는 32C1 (one-hot map)
        # Visualize할 때는 rgb8
        self.mask_pub = self.create_publisher(RosImage, self.mask_topic, 10)            
        self.mask_vis_pub = self.create_publisher(RosImage, self.mask_vis_topic, 10)      
        self.init_mask_pub = self.create_publisher(RosImage, self.init_mask_topic, 10) 
        self.init_mask_vis_pub = self.create_publisher(RosImage, self.init_mask_vis_topic, 10)   

        self.objects_to_track_sub = self.create_subscription(
            RosString, self.objects_to_track_topic, self._on_objects_to_track, 10
        )

        self.objects_to_track = self._parse_objects_to_track(args.objects_to_track)

        self.last_bgr = None
        self.last_header = None
        self.last_stamp_key = None
        self.last_mask_label = None
        self.init_mask_label = None

        # load GSAM once
        self.declare_parameter("gd_config", str(here / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"))
        self.declare_parameter("gd_ckpt", str(here / "groundingdino_swint_ogc.pth"))
        self.declare_parameter("sam_ckpt", str(here / "sam_vit_h_4b8939.pth"))
        gd_config = self.get_parameter("gd_config").value
        gd_ckpt = self.get_parameter("gd_ckpt").value
        sam_ckpt = self.get_parameter("sam_ckpt").value
        self.gd_model, self.gd_transform, self.sam_predictor, self.dev = load_gsam(
            gd_config, gd_ckpt, sam_ckpt, device=device
        )

        # load Cutie
        self.cutie = get_default_model().to(self.dev)
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.processor.max_internal_size = int(self.get_parameter("max_internal_size").value)

        self.inited = False
        hz = float(self.get_parameter("publish_rate").value)
        self.timer = self.create_timer(1.0 / max(hz, 1e-3), self._on_timer)

        self.get_logger().info(
            f"sub: {self.image_topic}, pub: {self.mask_topic, self.init_mask_topic}, "
            f"objects_to_track_topic: '{self.objects_to_track_topic}', objects={self.objects_to_track}, device: {self.dev}"
        )

    @staticmethod
    def _parse_objects_to_track(raw_text: str) -> list[str]:
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
    
    @staticmethod
    def _make_imgmsg_32sc1(label: np.ndarray, header) -> RosImage:
        """label: (H,W) int32, encoding=32SC1"""
        if label.dtype != np.int32:
            label = label.astype(np.int32)
        h, w = label.shape[:2]
        msg = RosImage()
        # IMPORTANT: stamp/frame_id must match the input image header.
        # Do not assign the header object directly (avoid accidental stamp drift).
        msg.header.stamp = header.stamp
        msg.header.frame_id = header.frame_id
        msg.height, msg.width = int(h), int(w)
        msg.encoding = "32SC1"
        msg.is_bigendian = False
        msg.step = int(w * 4)  # int32 = 4 bytes
        msg.data = label.tobytes()
        return msg

    @staticmethod
    def _make_imgmsg_rgb8(rgb: np.ndarray, header) -> RosImage:
        """rgb: (H,W,3) uint8, encoding=rgb8"""
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        h, w = rgb.shape[:2]
        msg = RosImage()
        # IMPORTANT: stamp/frame_id must match the input image header.
        msg.header.stamp = header.stamp
        msg.header.frame_id = header.frame_id
        msg.height = int(h)
        msg.width = int(w)
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = int(w * 3)
        msg.data = rgb.tobytes()
        return msg

    def _on_image(self, msg: RosImage):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Copy header so that published mask messages can carry the *input image* stamp,
        # independent of when this node actually publishes the mask.
        hdr = RosHeader()
        hdr.stamp = msg.header.stamp
        hdr.frame_id = msg.header.frame_id
        with self._img_lock:
            self.last_header = hdr
            self.last_bgr = bgr

    def _on_objects_to_track(self, msg: RosString):
        objs = self._parse_objects_to_track(msg.data)
        self.objects_to_track = objs
        self.get_logger().info(f"objects_to_track updated: {self.objects_to_track}")
        self.inited = False
        self.last_stamp_key = None

        self.last_mask_label = None
        self.init_mask_label = None

        # 2) Cutie 메모리 완전 리셋: InferenceCore 재생성
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.processor.max_internal_size = int(self.get_parameter("max_internal_size").value)


    @torch.inference_mode()
    def _on_timer(self):
        with self._img_lock:
            if self.last_bgr is None or self.last_header is None:
                return
            header = self.last_header
            bgr = self.last_bgr.copy()

        stamp = header.stamp
        stamp_key = (int(stamp.sec), int(stamp.nanosec))

        # 새 프레임이 없으면 마지막 마스크를 재-publish(주기 유지)
        if stamp_key == self.last_stamp_key and self.last_mask_label is not None:
            self._publish_label_and_vis(self.last_mask_label, header, self.mask_pub, self.mask_vis_pub)
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        image_tensor = torch.from_numpy(rgb).to(self.dev).permute(2, 0, 1).contiguous().float() / 255.0

        amp = (self.dev.type == "cuda")
        with torch.cuda.amp.autocast(enabled=amp):
            if not self.inited:
                if not self.objects_to_track:
                    return

                init_mask = np.zeros(rgb.shape[:2], dtype=np.int32)
                for instance_id, obj_name in enumerate(self.objects_to_track, start=1):
                    single_mask = gsam_make_mask(
                        bgr, obj_name, self.gd_model, self.gd_transform,
                        self.sam_predictor, self.dev, self.args.topk,
                        box_threshold=self.box_th, text_threshold=self.text_th,
                    )
                    init_mask[single_mask > 0] = instance_id
                init_mask = init_mask.astype(np.int32)
                self.init_mask_label = init_mask
                self._publish_label_and_vis(self.init_mask_label, header, self.init_mask_pub, self.init_mask_vis_pub)

                if init_mask.sum() == 0:
                    self.get_logger().warning("GSAM init mask is empty. (objects_to_track/threshold 확인)")
                    self.last_stamp_key = stamp_key
                    return

                init_mask_tensor = torch.from_numpy(init_mask).to(self.dev)
                obj_ids = list(range(1, int(init_mask.max()) + 1))

                output_prob = self.processor.step(image_tensor, init_mask_tensor, objects=obj_ids)
                self.inited = True
            else:
                output_prob = self.processor.step(image_tensor)
                if self.init_mask_label is not None:
                    self._publish_label_and_vis(self.init_mask_label, header,
                                                self.init_mask_pub, self.init_mask_vis_pub)

            cutie_mask = self.processor.output_prob_to_mask(output_prob)  # (H,W), 0/1/...
            cutie_mask_label = cutie_mask.detach().cpu().numpy().astype(np.int32)

        self.last_stamp_key = stamp_key
        self.last_mask_label = cutie_mask_label
        self._publish_label_and_vis(self.last_mask_label, header, self.mask_pub, self.mask_vis_pub)


    def _publish_mask(self, mask_bin_u8: np.ndarray, header):
        msg = self.bridge.cv2_to_imgmsg(mask_bin_u8, encoding="rgb8")
        # Keep stamp/frame_id aligned with the input image.
        msg.header.stamp = header.stamp
        msg.header.frame_id = header.frame_id
        self.mask_pub.publish(msg)

    def _publish_label_and_vis(self, label: np.ndarray, header, pub_label, pub_vis: Optional[any] = None):
        """Always publish label(32SC1). If pub_vis provided, also publish rgb8 visualization."""
        pub_label.publish(self._make_imgmsg_32sc1(label, header))
        if pub_vis is not None:
            rgb = label_to_rgb(label.astype(np.int32))
            pub_vis.publish(self._make_imgmsg_rgb8(rgb, header))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objects_to_track",
        type=str,
        default="",
        help="objects to track; supports JSON/python list string, comma-separated, or single object",
    )
    parser.add_argument("--topk", type=int, default=1, help="DINO topk initial box")
    args, _ = parser.parse_known_args()    

    rclpy.init()
    node = GSAMCutieTracker(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
