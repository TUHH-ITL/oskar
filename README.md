# Tipard UR20 ROS 2 Workspace

This workspace contains the ROS 2 packages needed to visualize and control the Tipard mobile base with a UR20 arm and Robotiq 2F-85 gripper through MoveIt 2.

Tested in this workspace with ROS 2 Humble.

## Isaac Sim Scene

Most commands in this README expect topics from the Isaac Sim scene. For simulation in Isaac Sim, open `robot_and_orchard.usd`. This scene has all ROS 2 OmniGraphs configured. The USD file is included in this repository.

The MoveIt ros2_control xacro currently uses an Isaac Sim topic bridge through:

```xml
<plugin>topic_based_ros2_control/TopicBasedSystem</plugin>
```

with these topics:

```text
/isaac_joint_commands
/isaac_joint_states
```

## Create a Workspace

```bash
mkdir -p ~/tipard_ws/src
cd ~/tipard_ws/src
```

## Packages Included

Copy/clone these source packages into the `src/` directory of a ROS 2 workspace:

```text
oskar_mapping
oskar_msgs
tipard_control
tipard_ur20_combined
tipard_ur20_moveit_config
```

`oskar_msgs` is required by `oskar_mapping` for the custom mapping messages. If you only want the Tipard base, UR20 MoveIt, and joystick launch files, `oskar_mapping` and `oskar_msgs` are optional.

No additional local source packages are required for the Tipard base, UR20 MoveIt, and joystick launch files. The runtime robot meshes are already inside `tipard_ur20_combined/meshes`, and the MoveIt package only depends on the local `tipard_ur20_combined` package.

The files `tipard_ur20_combined/create_combined_urdf.py` and `tipard_ur20_combined/fix_mesh_paths.py` are helper scripts from the URDF generation workflow. They contain references to the old source workspace and are not needed for normal build, visualization, MoveIt, or joystick operation.

## Clone External Packages

The ZED camera wrapper, ZED examples, and Fast Foundation Stereo are external repositories. Clone them only if you want the camera/mapping launches from the old workflow:

```bash
cd ~/tipard_ws/src
git clone https://github.com/stereolabs/zed-ros2-wrapper
git clone https://github.com/stereolabs/zed-ros2-examples
git clone https://github.com/NVlabs/Fast-FoundationStereo
```

These external packages are optional for the current Tipard/UR20 MoveIt bringup. They are required for:

- `ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedx sim_mode:=true use_sim_time:=true`
- `ros2 launch zed_display_rviz2 display_zed_cam.launch.py camera_model:=zedx sim_mode:=true use_sim_time:=true`

`Fast-FoundationStereo` is also required by the `oskar_mapping` disparity node.

## ROS Dependencies

Install ROS 2 Humble and MoveIt 2 first. Then install the runtime dependencies:

```bash
sudo apt update
sudo apt install \
  ros-humble-ament-cmake \
  ros-humble-controller-manager \
  ros-humble-geometry-msgs \
  ros-humble-joint-state-broadcaster \
  ros-humble-joint-state-publisher \
  ros-humble-joint-state-publisher-gui \
  ros-humble-joint-trajectory-controller \
  ros-humble-joy \
  ros-humble-launch \
  ros-humble-launch-ros \
  ros-humble-message-filters \
  ros-humble-moveit \
  ros-humble-moveit-msgs \
  ros-humble-moveit-servo \
  ros-humble-position-controllers \
  ros-humble-rclpy \
  ros-humble-robot-state-publisher \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-rosidl-default-generators \
  ros-humble-rosidl-default-runtime \
  ros-humble-rviz2 \
  ros-humble-sensor-msgs \
  ros-humble-std-msgs \
  ros-humble-std-srvs \
  ros-humble-stereo-msgs \
  ros-humble-tf2-ros \
  ros-humble-tf2-geometry-msgs \
  ros-humble-topic-based-ros2-control \
  ros-humble-warehouse-ros-mongo \
  ros-humble-xacro
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
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
```

Note: `tipard_control/package.xml` does not currently declare all Python and launch dependencies used by its nodes. The apt list above includes those missing runtime dependencies explicitly.

`oskar_mapping` also imports Python packages such as OpenCV, NumPy, PyTorch, scikit-learn, Detectron2, and Fast Foundation Stereo. Install those according to the mapping environment you use.

## Build

From the workspace root:

```bash
cd ~/tipard_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

You can check whether the workspace was sourced correctly by listing available ROS 2 packages:

```bash
ros2 pkg list
```

If the Tipard packages appear in the list, the workspace was built and sourced successfully.

To avoid sourcing the workspace manually every time, add it to your shell startup file:

```bash
echo "source ~/tipard_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

For the Tipard/UR20 subset, the package set was verified with:

```bash
colcon build --packages-select tipard_ur20_combined tipard_ur20_moveit_config tipard_control
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

Base teleop publishes to `/cmd_vel` and `/steering_mode`.

- Hold `RB` as the base dead-man switch.
- Hold `LB` for precision base motion.
- Press `A` for front-steer mode.
- Press `B` for symmetric four-wheel-steering mode.
- Press `X` for crab mode.

Arm Servo publishes twist commands to `/servo_node/delta_twist_cmds`.

- Hold `LB` to control the arm.
- Hold `RT` together with `LB` for precision arm motion.
- Left stick vertical controls arm X.
- Left stick horizontal controls arm Y.
- Right stick vertical controls arm Z.

The arm only moves while `LB` is held down.

## ZED Camera And Mapping

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

## Standalone Fake Hardware

For standalone fake hardware demos, switch the hardware plugin in `tipard_ur20_moveit_config/config/tipard_ur20_combined.ros2_control.xacro` to `mock_components/GenericSystem` and rebuild.
