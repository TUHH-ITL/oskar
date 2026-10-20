"""Start the full OSKAR hardware skeleton (fake hardware)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_fake_hardware',
            default_value='true',
            description='Run the tool node without real hardware.'),

        Node(
            package='oskar_hardware',
            executable='tool_node',
            name='tool_node',
            output='screen',
            parameters=[{'use_fake_hardware': use_fake_hardware}],
        ),
        Node(
            package='oskar_hardware',
            executable='execute_approach_node',
            name='execute_approach_node',
            output='screen',
        ),
    ])
