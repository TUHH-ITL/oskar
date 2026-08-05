# Tipard UR20 ROS 2 Workspace

This workspace contains the ROS 2 packages needed to visualize and control the Tipard mobile base with a UR20 arm and Robotiq 2F-85 gripper through MoveIt 2.

Tested in this workspace with ROS 2 Jazzy on Ubuntu 24.04.

## Isaac Sim Scene

Most commands in this README expect topics from the Isaac Sim (version 6.0.1) scene. For simulation in Isaac Sim, open `robot_and_orchard.usd`. This scene has all ROS 2 OmniGraphs configured. The USD file is tracked by DVC inside `oskar_simulation`.

### Getting the USD File

Install DVC with SSH support if not already installed:

```bash
pip install 'dvc[ssh]'
```

Then pull the Isaac Sim assets from the `oskar_simulation` directory:

```bash
cd oskar_simulation
dvc pull isaac_sim.dvc
```

This requires SSH access to `nas.flowcean.me` with the `oskar` user. The file `robot_and_orchard.usd` will be placed inside `oskar_simulation/isaac_sim/`.

Every scene under `isaac_sim/` opens from a fresh clone with no other checkout on the machine — geometry, textures, HDRIs and materials are all vendored alongside the scenes, and every asset path is relative, so the workspace can be cloned to any location. The only remote references are NVIDIA's own Isaac 6.0 asset URLs for the grid floor and the ZED X camera body, which stream from S3 on demand.

## Create a Workspace

```bash
mkdir -p ~/tipard_ws/src
cd ~/tipard_ws/src
```

## Packages Included

Clone this repo in your `src` folder in the ros2 workspace.

`oskar_msgs` is required by `oskar_mapping` for the custom mapping messages. If you only want the Tipard base, UR20 MoveIt, and joystick launch files, `oskar_mapping` and `oskar_msgs` are optional.

No additional local source packages are required for the Tipard base, UR20 MoveIt, and joystick launch files. The runtime robot meshes are already inside `tipard_ur20_combined/meshes`, and the MoveIt package only depends on the local `tipard_ur20_combined` package.

## Clone External Packages

The ZED camera wrapper, ZED examples, and Fast Foundation Stereo are external repositories. Clone them only if you want the camera/mapping launches:

```bash
cd ~/tipard_ws/src
git clone https://github.com/stereolabs/zed-ros2-wrapper
git clone https://github.com/stereolabs/zed-ros2-examples
git clone https://github.com/NVlabs/Fast-FoundationStereo
```

These external packages are required for:

- `ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedx sim_mode:=true use_sim_time:=true`
- `ros2 launch zed_display_rviz2 display_zed_cam.launch.py camera_model:=zedx sim_mode:=true use_sim_time:=true`

`Fast-FoundationStereo` is also required by the `oskar_mapping` disparity node.

## ROS Dependencies

Install ROS 2 Jazzy and MoveIt 2 first. Then install the runtime dependencies:

```bash
sudo apt update
sudo apt install \
  ros-jazzy-ament-cmake \
  ros-jazzy-controller-manager \
  ros-jazzy-geometry-msgs \
  ros-jazzy-joint-state-broadcaster \
  ros-jazzy-joint-state-publisher \
  ros-jazzy-joint-state-publisher-gui \
  ros-jazzy-joint-trajectory-controller \
  ros-jazzy-joy \
  ros-jazzy-launch \
  ros-jazzy-launch-ros \
  ros-jazzy-message-filters \
  ros-jazzy-moveit \
  ros-jazzy-moveit-msgs \
  ros-jazzy-moveit-servo \
  ros-jazzy-position-controllers \
  ros-jazzy-rclpy \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-ros-testing \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-rosidl-default-generators \
  ros-jazzy-rosidl-default-runtime \
  ros-jazzy-rviz2 \
  ros-jazzy-sensor-msgs \
  ros-jazzy-std-msgs \
  ros-jazzy-std-srvs \
  ros-jazzy-stereo-msgs \
  ros-jazzy-tf2-ros \
  ros-jazzy-tf2-geometry-msgs \
  ros-jazzy-warehouse-ros-sqlite \
  ros-jazzy-xacro
```

`topic-based-ros2-control` has no Jazzy Debian package. The MoveIt config uses its `topic_based_ros2_control/TopicBasedSystem` hardware plugin to talk to Isaac Sim, so it must be built from source in the workspace:

```bash
cd ~/tipard_ws/src
git clone https://github.com/PickNikRobotics/topic_based_ros2_control.git
```

Install `rosdep` if it is not already installed:

```bash
sudo apt install python3-rosdep
```

Initialize `rosdep` once per machine if this has not been done before:

```bash
sudo rosdep init
rosdep update
```

Then ask `rosdep` to install dependencies declared in the package manifests:

```bash
cd ~/tipard_ws
rosdep install --from-paths src --ignore-src -r -y --rosdistro jazzy
```

`rosdep` does not cover everything — `tipard_control/package.xml` omits some of the Python and launch dependencies its nodes use, which is why the apt list above is explicit.

`oskar_mapping` also imports Python packages such as OpenCV, NumPy, PyTorch, scikit-learn, Detectron2, and Fast Foundation Stereo. Install those according to the mapping environment you use.

## Build

From the workspace root:

```bash
cd ~/tipard_ws
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

To build only the Tipard/UR20 subset:

```bash
colcon build --packages-select tipard_ur20_combined tipard_ur20_moveit_config tipard_control
```

To avoid sourcing the workspace manually every time, add it to your shell startup file:

```bash
echo "source ~/tipard_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Full Bringup

Start the complete Tipard/UR20 bringup with:

```bash
ros2 launch tipard_control tipard_full_bringup.launch.py
```

This launch file starts the base joystick teleop:

```bash
ros2 launch tipard_control tipard_joystick_teleop.launch.py
```

and the MoveIt demo:

```bash
ros2 launch tipard_ur20_moveit_config demo.launch.py
```

## Individual Launch Commands

Start the MoveIt demo with RViz:

```bash
ros2 launch tipard_ur20_moveit_config demo.launch.py
```

Start the Tipard base joystick teleop:

```bash
ros2 launch tipard_control tipard_joystick_teleop.launch.py
```

## Robot Arm Control

The robot arm can be moved from RViz by sending a goal pose in the MoveIt planning interface.

To control the arm in Cartesian `x`, `y`, and `z` directions using the joystick, first move the arm into a bent position. This helps avoid singularities. Then launch:

```bash
ros2 launch tipard_control arm_servo_joystick.launch.py
```

## Joystick Controls 

Base teleop publishes to `/cmd_vel` and `/steering_mode`. The buttons correspond to Logitech joystick, adjust if you have a different joystick.

- Hold `RB` as the base dead-man switch.
- Hold `LB` for precision base motion.
- Press `A` for front-steer mode.
- Press `B` for symmetric four-wheel-steering mode.
- Press `X` for crab mode.

Arm Servo publishes twist commands to `/servo_node/delta_twist_cmds`. The arm only moves while `LB` is held down.

- Hold `LB` to control the arm.
- Hold `RT` together with `LB` for precision arm motion.
- Left stick vertical controls arm X.
- Left stick horizontal controls arm Y.
- Right stick vertical controls arm Z.

## ZED Camera And Depth Estimation

### Isaac Sim ZED Extension

The scenes drive their stereo camera through the Stereolabs OmniGraph node `sl.sensor.camera.ZED_Camera`, which comes from an extension that is **not** bundled with Isaac Sim. Its version must match the Kit version exactly — Isaac Sim 6.0.1 runs Kit 110.1.2, which `zed-isaac-sim` **v5.2.0** targets. Older tags (v4.x, Kit 107.3) fail to compile against it.

```bash
git clone https://github.com/stereolabs/zed-isaac-sim.git ~/zed-isaac-sim
cd ~/zed-isaac-sim
git checkout v5.2.0
./build.sh -u
```

Use `-u` so the extension version lock is regenerated; a stale lock pins a nonexistent `omni.graph.action-1.130.0`. If you are rebuilding after switching Kit versions, delete `_build _compiler _repo` first, or stale headers get linked in.

Then make the extension discoverable and load it at startup:

```bash
ln -s ~/zed-isaac-sim/exts/sl.sensor.camera ~/isaacsim_6_0_1/extsUser/
isaac-sim.sh --enable sl.sensor.camera
```

Both steps are needed. `extsUser/` only puts the extension on the search path — without `--enable` (or ticking AUTOLOAD in *Window -> Extensions*) the scene still fails with `Could not find node type interface for 'sl.sensor.camera.ZED_Camera'`.

To receive the simulated stream with the ZED SDK you also need **SDK 5.4.1 or newer** (<https://www.stereolabs.com/developers/release>); earlier versions cannot decode what v5.2.0 sends. On Ubuntu 24.04 with the CUDA 12.8 toolkit:

```bash
sudo ./ZED_SDK_Ubuntu24_cuda12.8_tensorrt10.9_v5.4.1.zstd.run -- silent skip_cuda skip_drivers
```

The installer bundles its own `pyzed` wheel, which declares `numpy>=2.0` and will **silently upgrade NumPy in `~/.local/lib`**. NumPy 2 breaks `cv_bridge`, OpenCV and Numba here, so pin it back afterwards:

```bash
python3 -m pip install --user --break-system-packages numpy==1.26.4
```

`pyzed` 5.4 works against NumPy 1.26.4 despite the declared constraint.

### Launching

These commands require the optional ZED and mapping packages listed above.

Launch the ZED camera driver in simulation mode:

```bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedx sim_mode:=true use_sim_time:=true
```

Launch the ZED RViz display:

```bash
ros2 launch zed_display_rviz2 display_zed_cam.launch.py camera_model:=zedx sim_mode:=true use_sim_time:=true
```

Run Fast Foundation Stereo mapping:

```bash
ros2 launch oskar_mapping mapping_sim.launch.py
```

The mapping launch publishes:

```text
/stereo/depth
/stereo/disparity
/stereo/points
```
