#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32


class Joystick4WSWDTeleop(Node):
    def __init__(self):
        super().__init__('joystick_4wswd_teleop')

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.mode_pub = self.create_publisher(Int32, '/steering_mode', 10)

        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)

        self.MODE_FRONT_STEER = 0
        self.MODE_SYMMETRIC_4WS = 1
        self.MODE_CRAB = 2

        self.current_mode = self.MODE_FRONT_STEER
        self.prev_buttons = []

        # Axes
        self.LEFT_STICK_HORIZONTAL = 0
        self.LEFT_STICK_VERTICAL = 1
        self.RIGHT_STICK_HORIZONTAL = 2

        # Buttons
        self.BUTTON_X = 0
        self.BUTTON_A = 1
        self.BUTTON_B = 2
        self.BUTTON_LB = 4   # precision mode
        self.BUTTON_RB = 5   # dead man switch

        # Normal speed limits
        self.max_vx = 1.0   # m/s
        self.max_vy = 1.0   # m/s
        self.max_wz = 1.0   # rad/s

        # Precision scaling factors
        self.precision_vx_scale = 0.25
        self.precision_vy_scale = 0.25
        self.precision_wz_scale = 0.30

        self.deadzone = 0.10

        self.publish_mode(self.current_mode)

        self.get_logger().info('Joystick 4WSWD teleop node started.')
        self.get_logger().info('A = front_steer, B = symmetric_4ws, X = crab')
        self.get_logger().info('RB = dead man, LB = precision mode')

    def apply_deadzone(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def publish_mode(self, mode: int):
        msg = Int32()
        msg.data = mode
        self.mode_pub.publish(msg)

        mode_name = {
            self.MODE_FRONT_STEER: 'front_steer',
            self.MODE_SYMMETRIC_4WS: 'symmetric_4ws',
            self.MODE_CRAB: 'crab'
        }.get(mode, 'unknown')

        self.get_logger().info(f'Switched mode to: {mode_name}')

        # Publish zero cmd_vel on mode switch for safety
        zero = Twist()
        self.cmd_vel_pub.publish(zero)

    def joy_callback(self, msg: Joy):
        if not self.prev_buttons:
            self.prev_buttons = [0] * len(msg.buttons)

        # ----------------------------
        # Mode switching on button edge
        # ----------------------------
        x_pressed = (msg.buttons[self.BUTTON_X] == 1 and self.prev_buttons[self.BUTTON_X] == 0)
        a_pressed = (msg.buttons[self.BUTTON_A] == 1 and self.prev_buttons[self.BUTTON_A] == 0)
        b_pressed = (msg.buttons[self.BUTTON_B] == 1 and self.prev_buttons[self.BUTTON_B] == 0)

        if a_pressed:
            self.current_mode = self.MODE_FRONT_STEER
            self.publish_mode(self.current_mode)

        elif b_pressed:
            self.current_mode = self.MODE_SYMMETRIC_4WS
            self.publish_mode(self.current_mode)

        elif x_pressed:
            self.current_mode = self.MODE_CRAB
            self.publish_mode(self.current_mode)

        # ----------------------------
        # Safety / modifiers
        # ----------------------------
        deadman_pressed = (msg.buttons[self.BUTTON_RB] == 1)
        precision_pressed = (msg.buttons[self.BUTTON_LB] == 1)

        # ----------------------------
        # Read joystick axes
        # ----------------------------
        left_x = self.apply_deadzone(msg.axes[self.LEFT_STICK_HORIZONTAL])
        left_y = self.apply_deadzone(msg.axes[self.LEFT_STICK_VERTICAL])
        right_x = self.apply_deadzone(msg.axes[self.RIGHT_STICK_HORIZONTAL])

        # ----------------------------
        # Choose speed scales
        # ----------------------------
        if precision_pressed:
            vx_scale = self.max_vx * self.precision_vx_scale
            vy_scale = self.max_vy * self.precision_vy_scale
            wz_scale = self.max_wz * self.precision_wz_scale
        else:
            vx_scale = self.max_vx
            vy_scale = self.max_vy
            wz_scale = self.max_wz

        # ----------------------------
        # Build Twist command
        # ----------------------------
        cmd = Twist()

        if deadman_pressed:
            if self.current_mode == self.MODE_FRONT_STEER:
                cmd.linear.x = vx_scale * left_y
                cmd.linear.y = 0.0
                cmd.angular.z = wz_scale * right_x

            elif self.current_mode == self.MODE_SYMMETRIC_4WS:
                cmd.linear.x = vx_scale * left_y
                cmd.linear.y = 0.0
                cmd.angular.z = wz_scale * right_x

            elif self.current_mode == self.MODE_CRAB:
                cmd.linear.x = vx_scale * left_y
                cmd.linear.y = vy_scale * left_x
                cmd.angular.z = 0.0

        else:
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd)

        self.prev_buttons = list(msg.buttons)


def main(args=None):
    rclpy.init(args=args)
    node = Joystick4WSWDTeleop()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        zero = Twist()
        node.cmd_vel_pub.publish(zero)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
