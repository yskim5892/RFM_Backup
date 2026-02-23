#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/bi_admin/RFM"
SESSION="rfm"

if [[ ! -f "$ROOT_DIR/rfm/install_humble/local_setup.bash" ]]; then
  echo "Missing $ROOT_DIR/rfm/install_humble/local_setup.bash"
  echo "Run first: source $ROOT_DIR/scripts/run_outside_humble.sh"
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
  "$COMMON && ros2 launch $ROOT_DIR/rfm/launch/rs.launch.py wrist_cam:=true" C-m

tmux new-window -t "$SESSION" -n "ur_driver"
tmux send-keys -t "$SESSION:ur_driver" \
  "$COMMON && ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=192.168.0.43 launch_rviz:=false" C-m

tmux new-window -t "$SESSION" -n "ur5_bridge"
tmux send-keys -t "$SESSION:ur5_bridge" \
  "$COMMON && python ur5_bridge.py --node_name ur5" C-m


RFM_ROOT="$(cd "$(dirname "$0")" && pwd)"
tmux new-window -t "$SESSION" -n "gello_robot"
tmux send-keys  -t "$SESSION:gello_robot" \
  "cd $RFM_ROOT/gello_software && source .venv/bin/activate && source /opt/ros/humble/setup.bash 2>/dev/null; python experiments/launch_nodes.py --robot ur --robot_ip 192.168.0.43 --hostname 0.0.0.0 --robot-port 6001" C-m

tmux select-window -t "$SESSION:realsense"
tmux attach -t "$SESSION"
