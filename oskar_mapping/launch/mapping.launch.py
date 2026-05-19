#!/usr/bin/env python3
"""Real robot launch file for Phase 1 mapping pipeline.

Loads real robot parameters and launches all 9 nodes.
Records key topics to rosbag.
"""

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for real robot mapping."""
    # Get package share directory
    oskar_mapping_share = FindPackageShare("oskar_mapping")
    config_dir = PathJoinSubstitution([oskar_mapping_share, "config"])

    # Parameter file (real robot)
    params_file = PathJoinSubstitution([config_dir, "mapping_params.yaml"])

    # Mapping nodes
    nodes = [
        Node(
            package="oskar_mapping",
            executable="stereo_sync_node",
            name="stereo_sync_node",
            parameters=[params_file],
            output="screen",
        ),
        Node(
            package="oskar_mapping",
            executable="disparity_node",
            name="disparity_node",
            parameters=[params_file],
            output="screen",
        ),
        Node(
            package="oskar_mapping",
            executable="segmentation_node",
            name="segmentation_node",
            parameters=[params_file],
            output="screen",
        ),
        Node(
            package="oskar_mapping",
            executable="depth_fusion_node",
            name="depth_fusion_node",
            parameters=[params_file],
            output="screen",
        ),
        Node(
            package="oskar_mapping",
            executable="backprojection_node",
            name="backprojection_node",
            parameters=[params_file],
            output="screen",
        ),
        Node(
            package="oskar_mapping",
            executable="multiview_fusion_node",
            name="multiview_fusion_node",
            parameters=[params_file],
            output="screen",
        ),
        Node(
            package="oskar_mapping",
            executable="bio_sanity_node",
            name="bio_sanity_node",
            parameters=[params_file],
            output="screen",
        ),
        Node(
            package="oskar_mapping",
            executable="tree_assignment_node",
            name="tree_assignment_node",
            parameters=[params_file],
            output="screen",
        ),
        Node(
            package="oskar_mapping",
            executable="thinning_decision_node",
            name="thinning_decision_node",
            parameters=[params_file],
            output="screen",
        ),
    ]

    # Rosbag recording
    rosbag_record = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "/stereo/sync_pair",
            "/flowers/masks",
            "/stereo/depth",
            "/flowers/detections_3d",
            "/flowers/world_obs",
            "/flowers/landmarks",
            "/flowers/landmarks_bio",
            "/orchard/per_tree_flowers",
            "/orchard/thinning_plan",
            "/tf",
            "/tf_static",
            "-o",
            "oskar_mapping_real",
        ],
        output="screen",
    )

    return LaunchDescription(nodes)  # + [rosbag_record])
