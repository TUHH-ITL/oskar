#!/usr/bin/env python3
"""Fake end-effector tool controller for the OSKAR thinning robot.

Serves the SetToolState service. In fake mode every command is simply logged,
delayed a little and reported as successful. When real hardware arrives, only
the three _hw_* methods below need a real implementation -- the service
interface, the state machine and every node above stay untouched.
"""

import time

import rclpy
from rclpy.node import Node

from oskar_msgs.srv import SetToolState

# Human readable names for logging.
COMMAND_NAMES = {
    SetToolState.Request.CMD_STOP: 'STOP',
    SetToolState.Request.CMD_PREPARE: 'PREPARE',
    SetToolState.Request.CMD_REMOVE: 'REMOVE',
}


class ToolNode(Node):
    """Tool abstraction: intent in, success/message out."""

    def __init__(self):
        super().__init__('tool_node')

        # Flip this to False once a real driver is wired in.
        self.declare_parameter('use_fake_hardware', True)
        # Simulated durations [s] so feedback timing looks realistic.
        self.declare_parameter('prepare_duration', 0.5)
        self.declare_parameter('remove_duration', 1.5)

        self._fake = self.get_parameter('use_fake_hardware').value
        self._state = SetToolState.Request.CMD_STOP
        self._removals = 0

        self._srv = self.create_service(
            SetToolState, 'set_tool_state', self.on_set_tool_state)

        self.get_logger().info(
            'tool_node ready on /set_tool_state '
            f'(use_fake_hardware={self._fake})')

    # ------------------------------------------------------------------
    # Service callback
    # ------------------------------------------------------------------
    def on_set_tool_state(self, request, response):
        name = COMMAND_NAMES.get(request.command)

        if name is None:
            response.success = False
            response.message = f'unknown command {request.command}'
            self.get_logger().error(response.message)
            return response

        self.get_logger().info(f'command {name} received')

        if not self._fake:
            response.success = False
            response.message = 'real hardware driver not implemented yet'
            self.get_logger().error(response.message)
            return response

        if request.command == SetToolState.Request.CMD_STOP:
            self._hw_stop()
            self._state = SetToolState.Request.CMD_STOP
            response.success = True
            response.message = 'tool stopped'

        elif request.command == SetToolState.Request.CMD_PREPARE:
            self._hw_prepare()
            self._state = SetToolState.Request.CMD_PREPARE
            response.success = True
            response.message = 'tool prepared'

        else:  # CMD_REMOVE
            # Deliberate guard: exercises the error path of every caller.
            if self._state != SetToolState.Request.CMD_PREPARE:
                response.success = False
                response.message = 'tool not prepared, send CMD_PREPARE first'
                self.get_logger().warning(response.message)
                return response

            self._hw_remove()
            self._removals += 1
            # After a removal the tool must be prepared again.
            self._state = SetToolState.Request.CMD_STOP
            response.success = True
            response.message = f'removal {self._removals} completed'

        self.get_logger().info(response.message)
        return response

    # ------------------------------------------------------------------
    # Hardware layer -- replace these three with the real driver calls
    # ------------------------------------------------------------------
    def _hw_stop(self):
        """Bring the tool into a safe, non-removing state."""
        pass

    def _hw_prepare(self):
        """Spin up / arm the removal mechanism."""
        time.sleep(self.get_parameter('prepare_duration').value)

    def _hw_remove(self):
        """Perform exactly one complete removal operation."""
        time.sleep(self.get_parameter('remove_duration').value)


def main(args=None):
    rclpy.init(args=args)
    node = ToolNode()
    try:
        # NOTE: the fake sleeps block this single-threaded executor. That is
        # fine for a stub; a real driver should either be non-blocking or run
        # in a MultiThreadedExecutor with a ReentrantCallbackGroup.
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
