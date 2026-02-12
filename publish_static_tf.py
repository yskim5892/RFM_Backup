import argparse
import numpy as np
from pathlib import Path

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

import math_utils  # must provide mat_to_quat(R): returns (qx, qy, qz, qw)


def build_T(vals):
    """vals: list[16] row-major -> 4x4 numpy array"""
    return np.array(vals, dtype=float).reshape(4, 4)


def load_T_from_file(matrix_file: str) -> np.ndarray:
    text = Path(matrix_file).read_text(encoding="utf-8")
    tokens = text.replace(",", " ").split()
    if len(tokens) != 16:
        raise ValueError(f"Matrix file must contain exactly 16 numbers, got {len(tokens)}: {matrix_file}")
    vals = [float(v) for v in tokens]
    return build_T(vals)


class StaticTFPublisher:
    def __init__(self, node: Node):
        self._node = node
        self._broadcaster = StaticTransformBroadcaster(node)

    def publish_from_matrix(self, T: np.ndarray, parent: str, child: str):
        # translation
        x, y, z = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])

        # rotation (3x3)
        R = T[0:3, 0:3]
        qx, qy, qz, qw = math_utils.mat_to_quat(R)

        t = TransformStamped()
        t.header.stamp = self._node.get_clock().now().to_msg()
        t.header.frame_id = parent
        t.child_frame_id = child

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z

        t.transform.rotation.x = float(qx)
        t.transform.rotation.y = float(qy)
        t.transform.rotation.z = float(qz)
        t.transform.rotation.w = float(qw)

        # publish once (static broadcaster uses transient local QoS)
        self._broadcaster.sendTransform(t)

        self._node.get_logger().info(
            f"Published static TF {parent} -> {child}: "
            f"t=({x:.6f},{y:.6f},{z:.6f}) "
            f"q=({qx:.6f},{qy:.6f},{qz:.6f},{qw:.6f})"
        )

    def publish_from_file(self, matrix_file: str, parent: str, child: str):
        T = load_T_from_file(matrix_file)
        self.publish_from_matrix(T=T, parent=parent, child=child)


def main():
    parser = argparse.ArgumentParser(
        description="Publish a static TF from a 4x4 matrix using rclpy/tf2_ros."
    )
    parser.add_argument(
        "m", nargs="*", type=float,
        help="(optional) 16 numbers for 4x4 transform matrix in row-major order"
    )
    parser.add_argument("--matrix-file", default="", help="Text file containing a 4x4 matrix (16 numbers)")
    parser.add_argument("--parent", default="tool0", help="parent frame (default: tool0)")
    parser.add_argument("--child", default="camera_link", help="child frame (default: camera_link)")
    parser.add_argument(
        "--once-and-exit",
        action="store_true",
        help="Publish once and exit immediately (default: keep node alive)."
    )
    args = parser.parse_args()

    if args.matrix_file:
        T = load_T_from_file(args.matrix_file)
    elif len(args.m) == 16:
        T = build_T(args.m)
    else:
        parser.error("Provide --matrix-file <path> or 16 matrix numbers.")

    rclpy.init()
    node = Node("static_tf_publisher_from_matrix")
    publisher = StaticTFPublisher(node)
    publisher.publish_from_matrix(T=T, parent=args.parent, child=args.child)

    try:
        if args.once_and_exit:
            # publish already done above; exit
            pass
        else:
            # keep alive so late-joining subscribers definitely get it, and for easier debugging
            rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
