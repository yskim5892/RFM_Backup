#!/usr/bin/env bash

ROOT_DIR="/home/bi_admin/RFM"
PKG_DIR="$ROOT_DIR/rfm"

_is_sourced() {
  [[ "${BASH_SOURCE[0]}" != "$0" ]]
}

_fail() {
  echo "$1" >&2
  if _is_sourced; then
    return 1
  fi
  exit 1
}

# Avoid chaining unrelated overlays into generated setup.bash.
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH

source /opt/ros/humble/setup.bash || _fail "Failed to source /opt/ros/humble/setup.bash"

colcon build \
  --packages-select rfm \
  --base-paths "$PKG_DIR" \
  --build-base "$PKG_DIR/build_humble" \
  --install-base "$PKG_DIR/install_humble" || _fail "colcon build failed"

source "$PKG_DIR/install_humble/local_setup.bash" || _fail "Failed to source $PKG_DIR/install_humble/local_setup.bash"

echo "[host] built and sourced: $PKG_DIR/install_humble/local_setup.bash"
