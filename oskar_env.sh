#!/usr/bin/env bash
# OSKAR environment setup — source this in EVERY terminal you use for OSKAR work:
#     source ~/ros2_ws/src/oskar/oskar_env.sh
#
# It sources ROS 2 Humble, the workspace overlay, and activates the isolated
# Python venv (~/oskar_venv) that has numpy<2 so ROS cv_bridge does not segfault.
# Your global numpy 2 / pyzed (ZED) stack is left untouched.

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/oskar_venv/bin/activate

echo "OSKAR env ready  |  python: $(which python3)  |  numpy: $(python3 -c 'import numpy; print(numpy.__version__)')"
