#!/usr/bin/env python3
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.ops import box_convert

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String as RosString
from cv_bridge import CvBridge

# GSAM = GroundingDINO + SAM
from groundingdino.util.inference import load_model, predict
from groundingdino.datasets import transforms as T
from segment_anything import sam_model_registry, SamPredictor

# Cutie (사용법은 사용자 제공/README 방식 그대로)
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

from utils import label_to_rgb

# -------------------------
# (2) GSAM 마스크 생성 함수들
# -------------------------
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
# (3) ROS2 노드: 첫 프레임 GSAM init → Cutie tracking → 15Hz publish
# -------------------------
class GSAMCutieTracker(Node):
    def __init__(self, args):
        super().__init__("gsam_cutie_tracker")
        self.args = args
        self.bridge = CvBridge()

        here = Path(__file__).resolve().parent / "thirdparty" / "Grounded-Segment-Anything"

        self.declare_parameter("image_topic", "/wrist_cam/camera/color/image_raw")
        self.declare_parameter("mask_topic", "/perception/instance_mask")
        self.declare_parameter("init_mask_topic", "/perception/initial_mask")
        self.declare_parameter("prompt", "/task/prompt")
        self.declare_parameter("publish_rate", 15.0)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("box_threshold", 0.35)
        self.declare_parameter("text_threshold", 0.25)
        self.declare_parameter("max_internal_size", 480)

        # GSAM paths (이 파일을 Grounded-Segment-Anything 루트에 두면 기본값 OK)
        self.declare_parameter(
            "gd_config",
            str(here / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"),
        )
        self.declare_parameter("gd_ckpt", str(here / "groundingdino_swint_ogc.pth"))
        self.declare_parameter("sam_ckpt", str(here / "sam_vit_h_4b8939.pth"))

        self.image_topic = self.get_parameter("image_topic").value
        self.mask_topic = self.get_parameter("mask_topic").value
        self.init_mask_topic = self.get_parameter("init_mask_topic").value
        self.prompt_topic = self.get_parameter("prompt").value

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
        self.mask_pub = self.create_publisher(RosImage, self.mask_topic, 10)
        self.init_mask_pub = self.create_publisher(RosImage, self.init_mask_topic, 10)

        # 나중에 외부에서 publish 구현. 일단 자체적으로 prompt를 argument로 받아 스스로 topic을 업데이트
        self.prompt_pub = self.create_publisher(RosString, "/task/prompt", 10)
        self.prompt_sub = self.create_subscription(RosString, "/task/prompt", self._on_prompt, 10)

        self.prompt = args.prompt
        if self.prompt:
            self.prompt_pub.publish(RosString(data=self.prompt))

        self.last_bgr = None
        self.last_header = None
        self.last_stamp_key = None
        self.last_mask = None  # uint8 0/255

        # load GSAM once
        gd_config = self.get_parameter("gd_config").value
        gd_ckpt = self.get_parameter("gd_ckpt").value
        sam_ckpt = self.get_parameter("sam_ckpt").value
        self.gd_model, self.gd_transform, self.sam_predictor, self.dev = load_gsam(
            gd_config, gd_ckpt, sam_ckpt, device=device
        )

        # load Cutie once
        self.cutie = get_default_model().to(self.dev)
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.processor.max_internal_size = int(self.get_parameter("max_internal_size").value)

        self.inited = False
        hz = float(self.get_parameter("publish_rate").value)
        self.timer = self.create_timer(1.0 / max(hz, 1e-3), self._on_timer)

        self.get_logger().info(
            f"sub: {self.image_topic}, pub: {self.mask_topic, self.init_mask_topic}, prompt: '{self.prompt_topic}', device: {self.dev}"
        )

    def _on_image(self, msg: RosImage):
        self.last_header = msg.header
        self.last_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _on_prompt(self, msg: RosString):
        self.prompt = msg.data
        self.inited = False
        self.last_stamp_key = None

        self.last_mask = None
        self.init_mask_msg = None

        # 2) Cutie 메모리 완전 리셋: InferenceCore 재생성
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.processor.max_internal_size = int(self.get_parameter("max_internal_size").value)


    @torch.inference_mode()
    def _on_timer(self):
        if self.last_bgr is None or self.last_header is None:
            return

        stamp = self.last_header.stamp
        stamp_key = (int(stamp.sec), int(stamp.nanosec))

        # 새 프레임이 없으면 마지막 마스크를 재-publish(주기 유지)
        if stamp_key == self.last_stamp_key and self.last_mask is not None:
            self._publish_mask(self.last_mask, self.last_header)
            return

        bgr = self.last_bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        image_tensor = torch.from_numpy(rgb).to(self.dev).permute(2, 0, 1).contiguous().float() / 255.0

        amp = (self.dev.type == "cuda")
        with torch.cuda.amp.autocast(enabled=amp):
            if not self.inited:
                init_mask = gsam_make_mask(
                    bgr, self.prompt, self.gd_model, self.gd_transform,
                    self.sam_predictor, self.dev, self.args.topk,
                    box_threshold=self.box_th, text_threshold=self.text_th,
                )
                self.init_mask_msg = self.bridge.cv2_to_imgmsg(label_to_rgb(init_mask), encoding='rgb8')
                self.init_mask_pub.publish(self.init_mask_msg)
                if init_mask.sum() == 0:
                    self.get_logger().warning("GSAM init mask is empty. (prompt/threshold 확인)")
                    self.last_stamp_key = stamp_key
                    return

                init_mask = (init_mask > 0).astype(np.uint8) * 1 
                init_mask_tensor = torch.from_numpy(init_mask).to(self.dev)

                obj_ids = list(range(1, int(init_mask.max()) + 1))
                output_prob = self.processor.step(image_tensor, init_mask_tensor, objects=obj_ids)
                self.inited = True
            else:
                output_prob = self.processor.step(image_tensor)
                self.init_mask_pub.publish(self.init_mask_msg)

            cutie_mask = self.processor.output_prob_to_mask(output_prob)  # (H,W), 0/1/...
            cutie_mask_rgb = label_to_rgb(cutie_mask.detach().cpu().numpy().astype(np.int32))

        self.last_stamp_key = stamp_key
        self.last_mask = cutie_mask_rgb
        self._publish_mask(cutie_mask_rgb, self.last_header)

    def _publish_mask(self, mask_bin_u8: np.ndarray, header):
        msg = self.bridge.cv2_to_imgmsg(mask_bin_u8, encoding="rgb8")
        msg.header = header
        self.mask_pub.publish(msg)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="", help="initial prompt string")
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

