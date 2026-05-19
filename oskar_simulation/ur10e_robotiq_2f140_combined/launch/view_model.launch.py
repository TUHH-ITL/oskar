from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_name = "ur10e_robotiq_2f140_combined"
    share_dir = Path(get_package_share_directory(package_name))
    urdf_path = share_dir / "urdf" / "ur10e_robotiq_2f140_combined.urdf"

    robot_description = urdf_path.read_text()
    use_gui = LaunchConfiguration("use_gui")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_gui",
                default_value="true",
                description="Use the joint_state_publisher_gui for manual joint control.",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": robot_description}],
                output="screen",
            ),
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                condition=IfCondition(use_gui),
                output="screen",
            ),
            Node(
                package="joint_state_publisher",
                executable="joint_state_publisher",
                condition=UnlessCondition(use_gui),
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                output="screen",
            ),
        ]
    )
