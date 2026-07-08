#!/usr/bin/env bash
# OSKAR environment setup — source this in EVERY terminal you use for OSKAR work:
#     source ~/ros2_ws/src/oskar/oskar_env.sh
#
# It sources ROS 2 Jazzy, the workspace overlay, and activates the isolated
# Python venv (~/oskar_venv) that has numpy<2 so ROS cv_bridge does not segfault.
# Your global numpy 2 / pyzed (ZED) stack is left untouched.

# Scrub any pre-existing Humble paths and ROS variables from the session to avoid pollution
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v 'humble' | paste -sd:)
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v 'humble' | paste -sd:)
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v 'humble' | paste -sd:)
unset ROS_DISTRO
unset ROS_VERSION
unset ROS_PYTHON_VERSION
unset AMENT_PREFIX_PATH

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/oskar_venv/bin/activate

echo "OSKAR env ready  |  python: $(which python3)  |  numpy: $(python3 -c 'import numpy; print(numpy.__version__)')"
