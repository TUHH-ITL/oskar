from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():

    servo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("tipard_ur20_moveit_config"),
                "launch",
                "servo.launch.py"
            ])
        )
    )

    arm_joystick_servo_node = Node(
        package="tipard_control",
        executable="arm_joystick_servo",
        name="arm_joystick_servo",
        output="screen"
    )

    start_servo = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "service",
                    "call",
                    "/servo_node/start_servo",
                    "std_srvs/srv/Trigger",
                    "{}"
                ],
                output="screen"
            )
        ]
    )

    translation_only_control = TimerAction(
        period=4.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "service",
                    "call",
                    "/servo_node/change_control_dimensions",
                    "moveit_msgs/srv/ChangeControlDimensions",
                    "{control_x_translation: true, control_y_translation: true, control_z_translation: true, control_x_rotation: false, control_y_rotation: false, control_z_rotation: false}"
                ],
                output="screen"
            )
        ]
    )

    return LaunchDescription([
        servo_launch,
        arm_joystick_servo_node,
        start_servo,
        translation_only_control,
    ])
