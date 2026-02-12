#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/bi_admin/RFM"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

if [[ ! -f "$ROOT_DIR/rfm/install_foxy/local_setup.bash" ]]; then
  echo "Missing $ROOT_DIR/rfm/install_foxy/local_setup.bash"
  echo "Run once first: bash $ROOT_DIR/scripts/build_rfm_foxy_once.sh"
  exit 1
fi

sudo docker run --gpus all -it --rm \
  --net=host \
  -e FASTRTPS_DEFAULT_PROFILES_FILE="$ROOT_DIR/fastdds_disable_shm.xml" \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
  -e ROS_LOCALHOST_ONLY="$ROS_LOCALHOST_ONLY" \
  -e RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION" \
  -v "$ROOT_DIR":"$ROOT_DIR" \
  foundationpose_ros \
  bash -lc "source /opt/ros/foxy/setup.bash && source $ROOT_DIR/rfm/install_foxy/local_setup.bash && cd $ROOT_DIR && echo '[docker] ready. run pose tracker with:' && echo 'python pose_tracker_action_node.py --ros-args -p bridge_node_name:=ur5' && echo \"[docker] env: ROS_DOMAIN_ID=$ROS_DOMAIN_ID ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION\" && exec bash"
