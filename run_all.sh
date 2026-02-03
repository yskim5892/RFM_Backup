#!/usr/bin/env bash
set -euo pipefail

SESSION="rfm"

# 인자로 받은 prompt 옵션 전체를 그대로 사용
# 예: --prompt apple
PROMPT_ARGS="${*:-"--prompt apple"}"

# 이미 세션이 있으면 안내 후 종료
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[!] tmux session '$SESSION' already exists."
  echo "    attach: tmux attach -t $SESSION"
  echo "    kill  : tmux kill-session -t $SESSION"
  exit 1
fi

COMMON_SETUP='
set -e
# source /opt/ros/humble/setup.bash
# source ~/RFM/rfm/install/setup.bash
'

tmux new-session -d -s "$SESSION" -n "realsense"
tmux send-keys  -t "$SESSION:realsense" \
  "$COMMON_SETUP ros2 launch rfm rs.launch.py wrist_cam:=true" C-m

tmux new-window -t "$SESSION" -n "ur_driver"
tmux send-keys  -t "$SESSION:ur_driver" \
  "$COMMON_SETUP ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=192.168.0.43 launch_rviz:=false" C-m

tmux new-window -t "$SESSION" -n "gsam_cutie"
tmux send-keys  -t "$SESSION:gsam_cutie" \
  "$COMMON_SETUP python gsam_cutie_tracker_node.py ${PROMPT_ARGS}" C-m

tmux new-window -t "$SESSION" -n "static_tf"
tmux send-keys  -t "$SESSION:static_tf" \
  "$COMMON_SETUP ros2 run tf2_ros static_transform_publisher -0.04 0.09 0.03 0 -0.25881905 0.96592583 0 tool0 camera_color_optical_frame" C-m

tmux select-window -t "$SESSION:realsense"
tmux attach -t "$SESSION"

