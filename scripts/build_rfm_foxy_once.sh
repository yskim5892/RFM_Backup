#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/bi_admin/RFM"
PKG_DIR="$ROOT_DIR/rfm"

sudo docker run --gpus all -it --rm \
  --net=host \
  -v "$ROOT_DIR":"$ROOT_DIR" \
  foundationpose_ros \
  bash -lc "source /opt/ros/foxy/setup.bash && colcon build --packages-select rfm --base-paths $PKG_DIR --build-base $PKG_DIR/build_foxy --install-base $PKG_DIR/install_foxy"

echo "[docker] built: $PKG_DIR/install_foxy/local_setup.bash"
