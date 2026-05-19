from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("ur10e_robotiq_2f140", package_name="ur10e_robotiq_2f140_moveit_config").to_moveit_configs()
    launch_description = generate_demo_launch(moveit_config)
    launch_description.add_action(
        Node(
            package="ur10e_robotiq_2f140_moveit_config",
            executable="add_ground_plane.py",
            output="screen",
        )
    )
    return launch_description
