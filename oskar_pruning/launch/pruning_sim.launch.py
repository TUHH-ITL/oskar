#!/usr/bin/env python3
"""Isaac Sim launch file for Phase 2 pruning pipeline."""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for Phase 2 pruning (sim)."""
    oskar_pruning_share = FindPackageShare("oskar_pruning")
    config_dir = PathJoinSubstitution([oskar_pruning_share, "config"])
    params_file = PathJoinSubstitution([config_dir, "pruning_params_sim.yaml"])

    nodes = [
        Node(
            package="oskar_pruning",
            executable="local_refinement_node",
            name="local_refinement_node",
            parameters=[params_file],
            output="screen",
        ),
        Node(
            package="oskar_pruning",
            executable="pruning_decision_node",
            name="pruning_decision_node",
            parameters=[params_file],
            output="screen",
        ),
    ]

    return LaunchDescription(nodes)
