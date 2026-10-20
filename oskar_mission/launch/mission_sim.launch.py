#!/usr/bin/env python3
"""The whole pruning phase against the Isaac Sim orchard.

thin:=false   drive and scan each tree; the arm is commanded directly.
thin:=true    also decide and remove: brings up the thinning decision and the
              MoveIt stack, and hands the scan to the arm's trajectory
              controller so only one thing commands the arm.
"""

from pathlib import Path

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def share(package, *parts):
    return str(Path(get_package_share_directory(package), *parts))


def launch_setup(context):
    thin = context.launch_configurations["thin"] == "true"
    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                share("oskar_online_perception", "launch", "perception_sim.launch.py")),
            launch_arguments={
                "scan_command_mode": "trajectory" if thin else "joint_commands",
                "rviz": context.launch_configurations["rviz"],
            }.items()),
        Node(package="oskar_mission", executable="mission_node", name="mission_node",
             parameters=[share("oskar_mission", "config", "mission_sim.yaml"),
                         {"thin_trees": thin}],
             output="screen"),
    ]
    if thin:
        actions += [
            Node(package="oskar_thinning_decision", executable="thinning_decision_node",
                 name="thinning_decision_node", parameters=[{"use_sim_time": True}],
                 output="screen"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    share("oskar_removal", "launch", "removal_sim.launch.py")),
                launch_arguments={
                    "hardware": "isaac",
                    "local_scene": context.launch_configurations["local_scene"],
                }.items()),
        ]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("thin", default_value="false",
                              description="decide and remove after each scan"),
        DeclareLaunchArgument("local_scene", default_value="false",
                              description="rebuild a target-carved octomap before each approach"),
        DeclareLaunchArgument("rviz", default_value="false"),
        OpaqueFunction(function=launch_setup),
    ])
