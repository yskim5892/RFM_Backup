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
  "$COMMON_SETUP python publish_static_tf.py 0 1 0 -0.04 -0.5 0 0.866 0.09 0.866 0 0.5 0.03 0 0 0 1" C-m

tmux new-window -t "$SESSION" -n "foundation_pose"
tmux send-keys -t "SESSION:foundation_pose" \
  "$COMMON_SETUP python pose_tracker_action_node.py --ros-args \
  -r __ns:=/pose_tracker \
  -r /pose_tracker/tf:=/tf \
  -r /pose_tracker/tf_static:=/tf_static \
  -p foundationpose_root:=/home/bi_admin/RFM/thirdparty/FoundationPose \
  -p mesh_file:=/home/bi_admin/RFM/ycb/013_apple/google_16k/textured.obj \
  -p base_frame:=base_link" C-m

tmux select-window -t "$SESSION:realsense"
tmux attach -t "$SESSION"

