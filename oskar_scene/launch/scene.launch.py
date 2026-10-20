from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    params = PathJoinSubstitution([
        FindPackageShare('oskar_scene'), 'config', 'scene.yaml'])
    return LaunchDescription([
        Node(
            package='oskar_scene',
            executable='local_scene_node',
            name='local_scene_node',
            namespace='scene',
            output='screen',
            parameters=[params]),
    ])
