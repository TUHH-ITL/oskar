from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    camera_name = LaunchConfiguration('camera_name')
    return LaunchDescription([
        DeclareLaunchArgument('camera_name', default_value='zed'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('zed_wrapper'), 'launch', 'zed_camera.launch.py'])),
            launch_arguments={
                'camera_model': 'zedx',
                'camera_name': camera_name,
            }.items()),
        Node(
            package='oskar_sensing',
            executable='snapshot_node',
            name='snapshot_node',
            namespace='sensing',
            output='screen',
            parameters=[{
                'source_cloud_topic': [
                    '/', camera_name, '/zed_node/point_cloud/cloud_registered'],
                'output_cloud_topic': '/scene/input_cloud',
            }]),
    ])
