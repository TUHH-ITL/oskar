#!/bin/zsh
# Capture one dataset split: bring the stack up, drive the sweeps, take it down.
#
#   capture_split.sh <log_dir> <dataset_root> <split> [sweep ...]
#
# Only this launch's own process group is stopped at the end, so a simulator or
# another run started elsewhere is left alone.
set -e
source /opt/ros/jazzy/setup.zsh
cd /home/workstation/ros2_ws
source install/setup.zsh
export VIRTUAL_ENV=/home/workstation/oskar_venv
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

LOG_DIR="$1"; ROOT="$2"; SPLIT="$3"

while pgrep -f "oskar_online_perception/lib" > /dev/null; do sleep 2; done

setsid ros2 launch oskar_online_perception perception_sim.launch.py capture:=true \
    capture_root:=$ROOT capture_split:=$SPLIT capture_stride:=3 \
    > $LOG_DIR/capture_$SPLIT.log 2>&1 &
LAUNCH=$!
GROUP=$(ps -o pgid= -p $LAUNCH | tr -d ' ')
sleep 25
/home/workstation/oskar_venv/bin/python3 \
    src/oskar/oskar_online_perception/scripts/drive_scan_sweeps.py ${@:4} \
    > $LOG_DIR/sweeps_$SPLIT.log 2>&1
sleep 10
kill -INT -$GROUP 2>/dev/null || true
sleep 5
echo "capture done for $SPLIT"
