#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/bi_admin/RFM"
SESSION="rfm"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

if [[ ! -f "$ROOT_DIR/rfm/install_humble/local_setup.bash" ]]; then
  echo "Missing $ROOT_DIR/rfm/install_humble/local_setup.bash"
  echo "Run first: source $ROOT_DIR/scripts/run_outside_humble.sh"
  exit 1
fi
if [[ ! -f "$ROOT_DIR/rfm/install_foxy/local_setup.bash" ]]; then
  echo "Missing $ROOT_DIR/rfm/install_foxy/local_setup.bash"
  echo "Run once first: bash $ROOT_DIR/scripts/build_rfm_foxy_once.sh"
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists."
  echo "attach: tmux attach -t $SESSION"
  echo "kill  : tmux kill-session -t $SESSION"
  exit 1
fi

COMMON="source /opt/ros/humble/setup.bash && source $ROOT_DIR/rfm/install_humble/local_setup.bash && cd $ROOT_DIR"

tmux new-session -d -s "$SESSION" -n "realsense"
tmux send-keys -t "$SESSION:realsense" \
  "export ROS_DOMAIN_ID=$ROS_DOMAIN_ID ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION && $COMMON && ros2 launch $ROOT_DIR/rfm/launch/rs.launch.py wrist_cam:=true" C-m

tmux new-window -t "$SESSION" -n "ur_driver"
tmux send-keys -t "$SESSION:ur_driver" \
  "export ROS_DOMAIN_ID=$ROS_DOMAIN_ID ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION && $COMMON && ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=192.168.0.43 launch_rviz:=false" C-m

tmux new-window -t "$SESSION" -n "ur5_bridge"
tmux send-keys -t "$SESSION:ur5_bridge" \
  "export ROS_DOMAIN_ID=$ROS_DOMAIN_ID ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION && $COMMON && python ur5_bridge.py --node_name ur5" C-m

tmux new-window -t "$SESSION" -n "gsam"
tmux send-keys -t "$SESSION:gsam" \
  "export ROS_DOMAIN_ID=$ROS_DOMAIN_ID ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION && $COMMON && echo '[gsam] example commands:' && echo \"ros2 topic pub -1 /inference/objects_to_track std_msgs/msg/String   \\\"{data: 'apple, lemon, baseball'}\\\"\" && echo \"ros2 topic pub -1 /inference/prompt std_msgs/msg/String \\\"{data: 'pick lemon'}\\\"\" && python gsam_cutie_tracker_node.py" C-m

tmux new-window -t "$SESSION" -n "pose_tracker"
tmux send-keys -t "$SESSION:pose_tracker" \
  "cd $ROOT_DIR && ROS_DOMAIN_ID=$ROS_DOMAIN_ID ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION bash scripts/run_inside_docker_foxy.sh" C-m

tmux select-window -t "$SESSION:realsense"
tmux attach -t "$SESSION"
