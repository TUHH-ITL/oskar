#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy
from geometry_msgs.msg import TwistStamped


class ArmJoystickServo(Node):
    def __init__(self):
        super().__init__('arm_joystick_servo')

        self.servo_pub = self.create_publisher(
            TwistStamped,
            '/servo_node/delta_twist_cmds',
            10
        )

        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )

        # Joystick axes
        self.LEFT_STICK_HORIZONTAL = 0
        self.LEFT_STICK_VERTICAL = 1
        self.RIGHT_STICK_VERTICAL = 3

        # Buttons
        self.BUTTON_LB = 4   # hold LB to control the arm
        self.BUTTON_RT = 7   # hold RT for precision mode

        # Servo command frame
        self.command_frame = 'ur20_base_link'

        # Normal arm speed.
        # Since servo.yaml uses command_in_type: "unitless",
        # these values should stay within [-1.0, 1.0].
        self.normal_scale = 1.0

        # Precision arm speed when RT is held.
        self.precision_scale = 0.30

        self.deadzone = 0.10

        self.get_logger().info('Arm joystick Servo node started.')
        self.get_logger().info('Hold LB / button 4 to move the arm.')
        self.get_logger().info('Hold RT / button 7 together with LB for precision mode.')
        self.get_logger().info('Left stick vertical = X, left stick horizontal = Y, right stick vertical = Z.')

    def apply_deadzone(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def joy_callback(self, msg: Joy):
        if len(msg.buttons) <= max(self.BUTTON_LB, self.BUTTON_RT):
            return

        arm_deadman_pressed = msg.buttons[self.BUTTON_LB] == 1
        precision_pressed = msg.buttons[self.BUTTON_RT] == 1

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = self.command_frame

        if arm_deadman_pressed:
            if len(msg.axes) > self.RIGHT_STICK_VERTICAL:
                scale = self.precision_scale if precision_pressed else self.normal_scale

                x = self.apply_deadzone(msg.axes[self.LEFT_STICK_VERTICAL])
                y = self.apply_deadzone(msg.axes[self.LEFT_STICK_HORIZONTAL])
                z = self.apply_deadzone(msg.axes[self.RIGHT_STICK_VERTICAL])

                cmd.twist.linear.x = scale * x
                cmd.twist.linear.y = scale * y
                cmd.twist.linear.z = scale * z

        # Always publish, including zero commands, so Servo stops cleanly.
        self.servo_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ArmJoystickServo()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = TwistStamped()
        stop.header.stamp = node.get_clock().now().to_msg()
        stop.header.frame_id = node.command_frame
        node.servo_pub.publish(stop)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
