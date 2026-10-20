#!/usr/bin/env python3
"""Online perception against the Isaac Sim orchard: detect, localize, map, scan.

Expects the scene patched by oskar_simulation/scripts/patch_scene_for_pruning.py
and playing: it supplies /clock, ground-truth TF for the robot, and the wrist
camera streams.
"""

import os

import yaml
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from pathlib import Path


def camera_static_transforms():
    """Static transforms for the camera optical frames, exported from the scene.

    Isaac publishes TF for articulation links only, so the frames the image
    headers name have no transform of their own.
    """
    config = Path(get_package_share_directory("oskar_online_perception")) / "config/camera_frames_sim.yaml"
    data = yaml.safe_load(config.read_text())["camera_frames"]
    parent = data["parent_frame"]
    nodes = []
    for frame, spec in data["frames"].items():
        t = spec["translation"]
        q = spec["rotation_xyzw"]
        nodes.append(Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name=f"{parent}_to_{frame}",
            arguments=["--x", str(t[0]), "--y", str(t[1]), "--z", str(t[2]),
                       "--qx", str(q[0]), "--qy", str(q[1]), "--qz", str(q[2]), "--qw", str(q[3]),
                       "--frame-id", parent, "--child-frame-id", frame],
            parameters=[{"use_sim_time": True}],
        ))
    return nodes


def node_interpreter():
    """The interpreter the nodes must run under, if one is active.

    The console scripts are built with a system-python shebang, but the models
    live in the workspace's virtual environment, which is where numpy is pinned
    below 2 for cv_bridge. Without this the perception node starts and then dies
    on `No module named detectron2` the moment a real detector is selected.
    """
    venv = os.environ.get("VIRTUAL_ENV")
    if not venv:
        return None
    interpreter = Path(venv) / "bin/python3"
    return str(interpreter) if interpreter.is_file() else None


def generate_launch_description():
    params = str(Path(get_package_share_directory("oskar_online_perception"))
                 / "config/perception_sim.yaml")
    capture = LaunchConfiguration("capture")

    common = dict(package="oskar_online_perception", parameters=[params], output="screen")
    interpreter = node_interpreter()
    if interpreter:
        common["prefix"] = interpreter

    return LaunchDescription([
        # trajectory hands the scan path to the arm's joint trajectory
        # controller; needed whenever oskar_removal's ros2_control stack is up.
        DeclareLaunchArgument("scan_command_mode", default_value="joint_commands",
                              description="joint_commands or trajectory"),
        # Capture mode records the renderer's own masks as a training set, so
        # the detector and the evaluation are left out: they only compete for
        # the GPU with the renderer that is producing the frames.
        DeclareLaunchArgument("rviz", default_value="false",
                              description="open RViz on the map, the markers and the overlay"),
        DeclareLaunchArgument("capture", default_value="false",
                              description="record frames and masks instead of detecting"),
        DeclareLaunchArgument("capture_root", default_value="",
                              description="dataset root for capture mode"),
        DeclareLaunchArgument("capture_split", default_value="train",
                              description="split name for capture mode"),
        DeclareLaunchArgument("capture_stride", default_value="3",
                              description="keep every Nth synchronised frame"),

        # Isaac's transform publisher names its root frame "world".
        Node(package="tf2_ros", executable="static_transform_publisher", name="map_to_world",
             arguments=["--frame-id", "map", "--child-frame-id", "world"],
             parameters=[{"use_sim_time": True}]),
        *camera_static_transforms(),

        Node(executable="detection_3d_node", name="detection_3d_node",
             condition=UnlessCondition(capture), **common),
        Node(executable="landmark_map_node", name="landmark_map_node",
             condition=UnlessCondition(capture), **common),
        Node(executable="gt_eval_node", name="gt_eval_node",
             condition=UnlessCondition(capture), **common),
        Node(executable="dataset_capture_node", name="dataset_capture_node",
             condition=IfCondition(capture),
             parameters=[params, {"out_root": LaunchConfiguration("capture_root"),
                                  "split": LaunchConfiguration("capture_split"),
                                  "stride": ParameterValue(LaunchConfiguration("capture_stride"),
                                            value_type=int)}],
             package="oskar_online_perception", output="screen",
             **({"prefix": interpreter} if interpreter else {})),
        Node(executable="scan_controller_node", name="scan_controller_node",
             package="oskar_online_perception", output="screen",
             parameters=[params, {"command_mode": LaunchConfiguration("scan_command_mode")}],
             **({"prefix": interpreter} if interpreter else {})),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             condition=IfCondition(LaunchConfiguration("rviz")),
             arguments=["-d", str(Path(get_package_share_directory("oskar_online_perception"))
                                  / "config/perception_sim.rviz")],
             parameters=[{"use_sim_time": True}], output="screen"),
    ])
