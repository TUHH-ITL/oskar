#!/usr/bin/env python3
"""Isaac Sim launch file for Phase 1 mapping pipeline.

Loads simulation parameters and launches all 9 nodes.
Records key topics to rosbag.
"""

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for Isaac Sim mapping."""
    # Get package share directory
    oskar_mapping_share = FindPackageShare("oskar_mapping")
    config_dir = PathJoinSubstitution([oskar_mapping_share, "config"])

    # Parameter file (simulation)
    params_file = PathJoinSubstitution([config_dir, "mapping_params_sim.yaml"])

    # Static transforms (only if Isaac Sim doesn't publish them)
    # These connect the cameras to the robot base
    static_tf_nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="map_to_odom",
            arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="odom_to_base_link",
            arguments=["0", "0", "0", "0", "0", "0", "odom", "base_link"],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_link_to_camera_1",
            # Adjust rotation/translation to match Isaac Sim camera mount
            arguments=[
                "0.1",
                "0.05",
                "0.5",
                "0",
                "0",
                "0",
                "base_link",
                "static_camera_1",
            ],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_link_to_camera_2",
            # Adjust rotation/translation to match Isaac Sim camera mount (stereo baseline ~0.15m)
            arguments=[
                "0.1",
                "-0.1",
                "0.5",
                "0",
                "0",
                "0",
                "base_link",
                "static_camera_2",
            ],
        ),
    ]

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
    # Record key topics for debugging and offline processing
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
            "oskar_mapping_sim",
        ],
        output="screen",
    )

    return LaunchDescription(static_tf_nodes + nodes)  # + [rosbag_record])
