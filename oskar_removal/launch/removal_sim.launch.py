#!/usr/bin/env python3
"""MoveIt, ros2_control and the removal nodes for the tipard + UR20.

hardware:=isaac   the arm is Isaac's, through topic_based_ros2_control on
                  /isaac_joint_commands; run alongside perception_sim.launch.py
                  with scan_controller_node's command_mode set to trajectory.
hardware:=mock    no simulator: mock_components holds the joints, for checking
                  planning and the action chain on their own.

The MoveIt config is tipard_ur20_moveit_config, unchanged on disk. Two things
are adjusted in memory: the hardware plugin for mock, and the SRDF's planar
virtual joint, which becomes floating. A planar joint drops the base's height
(1.045 m when settled), so MoveIt would place the robot a metre below the TF
the octomap and the goals are expressed in.
"""

from pathlib import Path

import xacro
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

MOCK_BASE_HEIGHT_M = "1.045"


def launch_setup(context):
    hardware = context.launch_configurations["hardware"]
    if hardware not in ("isaac", "mock"):
        raise RuntimeError(f"hardware must be isaac or mock, not {hardware}")
    sim_time = hardware == "isaac"
    local_scene = context.launch_configurations["local_scene"] == "true"

    moveit_share = Path(get_package_share_directory("tipard_ur20_moveit_config"))
    share = Path(get_package_share_directory("oskar_removal"))
    params = str(share / "config/removal_sim.yaml")

    urdf = xacro.process_file(
        str(moveit_share / "config/tipard_ur20_combined.urdf.xacro"),
        mappings={"initial_positions_file":
                  str(moveit_share / "config/initial_positions.yaml")}).toxml()
    if hardware == "mock":
        urdf = urdf.replace("topic_based_ros2_control/TopicBasedSystem",
                            "mock_components/GenericSystem")
    srdf = (moveit_share / "config/tipard_ur20_combined.srdf").read_text()
    srdf = srdf.replace('type="planar" parent_frame="world"', 'type="floating" parent_frame="world"')

    moveit_config = (
        MoveItConfigsBuilder("tipard_ur20_combined", package_name="tipard_ur20_moveit_config")
        .sensors_3d(str(share / "config/sensors_3d.yaml"))
        .to_moveit_configs())
    moveit_config.robot_description = {"robot_description": urdf}
    moveit_config.robot_description_semantic = {"robot_description_semantic": srdf}

    time = {"use_sim_time": sim_time}
    # Isaac publishes the robot's TF itself; a second publisher would fight it.
    rsp_remap = [("/tf", "tf_unused"), ("/tf_static", "tf_static_unused")] if sim_time else []

    nodes = [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[moveit_config.robot_description, time], remappings=rsp_remap,
             output="screen"),
        Node(package="controller_manager", executable="ros2_control_node",
             parameters=[str(moveit_share / "config/ros2_controllers.yaml"),
                         str(share / "config/controller_constraints.yaml"), time],
             remappings=[("/controller_manager/robot_description", "/robot_description")],
             output="screen"),
        Node(package="controller_manager", executable="spawner",
             arguments=["joint_state_broadcaster", "ur_manipulator_controller",
                        "--switch-timeout", "120", "--service-call-timeout", "120"],
             parameters=[time], output="screen"),
        Node(package="moveit_ros_move_group", executable="move_group",
             parameters=[moveit_config.to_dict(), time,
                         {"publish_robot_description_semantic": True,
                          "octomap_frame": "world", "octomap_resolution": 0.02}],
             output="screen"),
        Node(package="oskar_hardware", executable="tool_node", name="tool_node",
             parameters=[params, time], output="screen"),
        Node(package="oskar_hardware", executable="execute_approach_node",
             name="execute_approach_node", parameters=[params, time], output="screen"),
        Node(package="oskar_removal", executable="removal_planner_node",
             name="removal_planner_node", parameters=[params, time], output="screen"),
        Node(package="oskar_removal", executable="removal_executor_node",
             name="removal_executor_node",
             parameters=[params, time, {"use_local_scene": local_scene}], output="screen"),
    ]
    if hardware == "mock":
        nodes += [
            Node(package="tf2_ros", executable="static_transform_publisher",
                 arguments=["--z", MOCK_BASE_HEIGHT_M, "--frame-id", "world",
                            "--child-frame-id", "base_link"]),
            Node(package="tf2_ros", executable="static_transform_publisher",
                 arguments=["--frame-id", "map", "--child-frame-id", "world"]),
        ]
    if local_scene:
        nodes += [
            Node(package="oskar_scene", executable="local_scene_node", name="local_scene_node",
                 namespace="scene", parameters=[params, time], output="screen"),
            Node(package="oskar_scene", executable="depth_cloud_node", name="depth_cloud_node",
                 namespace="scene", parameters=[params, time], output="screen"),
        ]
    if context.launch_configurations["rviz"] == "true":
        nodes.append(Node(
            package="rviz2", executable="rviz2",
            arguments=["-d", str(moveit_share / "config/moveit.rviz")],
            parameters=[moveit_config.robot_description,
                        moveit_config.robot_description_semantic,
                        moveit_config.robot_description_kinematics, time],
            output="screen"))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("hardware", default_value="isaac",
                              description="isaac or mock"),
        DeclareLaunchArgument("local_scene", default_value="false",
                              description="rebuild a target-carved octomap before each approach"),
        DeclareLaunchArgument("rviz", default_value="false"),
        OpaqueFunction(function=launch_setup),
    ])
