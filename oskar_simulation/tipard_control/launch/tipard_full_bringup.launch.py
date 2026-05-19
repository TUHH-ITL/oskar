from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():

    joystick_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("tipard_control"),
                "launch",
                "tipard_joystick_teleop.launch.py"
            ])
        )
    )

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("tipard_ur20_moveit_config"),
                "launch",
                "demo.launch.py"
            ])
        )
    )

    return LaunchDescription([
        joystick_launch,
        moveit_launch
    ])