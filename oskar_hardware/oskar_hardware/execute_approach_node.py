#!/usr/bin/env python3
"""MoveIt-backed execution of a tool-frame approach motion."""

import copy
import math
import threading
import time

from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    Constraints,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from moveit_msgs.srv import GetCartesianPath
from oskar_msgs.action import ExecuteApproach
import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from shape_msgs.msg import SolidPrimitive
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener


class ExecuteApproachNode(Node):
    """Plan and execute stand-off and Cartesian approach motions."""

    def __init__(self):
        super().__init__('execute_approach_node')
        self._callbacks = ReentrantCallbackGroup()
        self._active_lock = threading.Lock()
        self._goal_active = False

        self.declare_parameter('planning_group', 'ur_manipulator')
        self.declare_parameter('planning_frame', 'base_link')
        self.declare_parameter('tool_link', 'tool0')
        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('server_timeout', 5.0)
        self.declare_parameter('execution_timeout', 60.0)
        self.declare_parameter('transform_timeout', 2.0)
        self.declare_parameter('position_tolerance', 0.002)
        self.declare_parameter('orientation_tolerance', 0.02)
        self.declare_parameter('velocity_scaling', 0.15)
        self.declare_parameter('acceleration_scaling', 0.15)
        self.declare_parameter('cartesian_step', 0.005)
        self.declare_parameter('minimum_cartesian_fraction', 0.99)
        # Goals are tool tip poses; the tip sits this far along tool_link's +Z.
        self.declare_parameter('tcp_offset', 0.0)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(
            self._tf_buffer, self, spin_thread=False)
        self._move_group = ActionClient(
            self, MoveGroup, 'move_action', callback_group=self._callbacks)
        self._cartesian_path = self.create_client(
            GetCartesianPath,
            'compute_cartesian_path',
            callback_group=self._callbacks)
        self._execute_trajectory = ActionClient(
            self,
            ExecuteTrajectory,
            'execute_trajectory',
            callback_group=self._callbacks)
        self._server = ActionServer(
            self,
            ExecuteApproach,
            'execute_approach',
            execute_callback=self.execute,
            goal_callback=self.on_goal,
            cancel_callback=self.on_cancel,
            callback_group=self._callbacks)

        self.get_logger().info('ready on /execute_approach')

    def on_goal(self, _goal_request):
        with self._active_lock:
            if self._goal_active:
                self.get_logger().warning(
                    'rejecting goal: an approach is already active')
                return GoalResponse.REJECT
            self._goal_active = True
        return GoalResponse.ACCEPT

    def on_cancel(self, _goal_handle):
        return CancelResponse.ACCEPT

    def execute(self, goal_handle):
        result = ExecuteApproach.Result()
        deadline = time.monotonic() + self._parameter('execution_timeout')

        try:
            validation_error = self._validate_goal(goal_handle.request)
            if validation_error:
                return self._abort(
                    goal_handle,
                    result,
                    ExecuteApproach.Result.ERROR_INVALID_GOAL,
                    validation_error)

            self._feedback(
                goal_handle,
                ExecuteApproach.Feedback.PHASE_TRANSFORMING,
                0.0,
                'transforming target pose')
            target_pose = self._tip_to_link(
                self._transform_target(goal_handle.request.target_pose))
            self._feedback(
                goal_handle,
                ExecuteApproach.Feedback.PHASE_TRANSFORMING,
                1.0,
                'target pose transformed')

            wait_error = self._wait_for_moveit(goal_handle, deadline)
            if wait_error:
                return self._wait_failure(goal_handle, result, wait_error)

            if goal_handle.request.motion == ExecuteApproach.Goal.MOTION_LINEAR:
                return self._linear(goal_handle, result, deadline, target_pose,
                                    goal_handle.request.approach_distance)

            move_goal = self._move_group_goal(target_pose)
            self._feedback(
                goal_handle,
                ExecuteApproach.Feedback.PHASE_PLANNING,
                0.0,
                'planning to stand-off pose')
            move_result, wait_error = self._run_action(
                self._move_group,
                move_goal,
                goal_handle,
                deadline,
                self._move_group_feedback(goal_handle))
            if wait_error:
                return self._wait_failure(goal_handle, result, wait_error)
            if move_result.error_code.val != MoveItErrorCodes.SUCCESS:
                return self._moveit_failure(
                    goal_handle, result, move_result.error_code,
                    'failed to reach stand-off pose')

            if goal_handle.request.approach_distance <= 1e-6:
                return self._succeed(goal_handle, result)

            self._feedback(
                goal_handle,
                ExecuteApproach.Feedback.PHASE_PLANNING_APPROACH,
                0.0,
                'planning Cartesian approach')
            cartesian_request = self._cartesian_request(
                target_pose, goal_handle.request.approach_distance)
            cartesian_response, wait_error = self._run_future(
                self._cartesian_path.call_async(cartesian_request),
                goal_handle,
                deadline)
            if wait_error:
                return self._wait_failure(goal_handle, result, wait_error)
            if cartesian_response.error_code.val != MoveItErrorCodes.SUCCESS:
                return self._moveit_failure(
                    goal_handle, result, cartesian_response.error_code,
                    'Cartesian planning failed')

            minimum_fraction = self._parameter('minimum_cartesian_fraction')
            if cartesian_response.fraction < minimum_fraction:
                message = (
                    f'Cartesian path fraction {cartesian_response.fraction:.3f} '
                    f'is below required {minimum_fraction:.3f}')
                return self._abort(
                    goal_handle,
                    result,
                    ExecuteApproach.Result.ERROR_CARTESIAN_INCOMPLETE,
                    message)

            execute_goal = ExecuteTrajectory.Goal()
            execute_goal.trajectory = cartesian_response.solution
            self._feedback(
                goal_handle,
                ExecuteApproach.Feedback.PHASE_APPROACHING,
                0.0,
                'executing Cartesian approach')
            execute_result, wait_error = self._run_action(
                self._execute_trajectory,
                execute_goal,
                goal_handle,
                deadline)
            if wait_error:
                return self._wait_failure(goal_handle, result, wait_error)
            if execute_result.error_code.val != MoveItErrorCodes.SUCCESS:
                return self._moveit_failure(
                    goal_handle, result, execute_result.error_code,
                    'Cartesian execution failed')

            return self._succeed(goal_handle, result)
        except TransformException as exc:
            return self._abort(
                goal_handle,
                result,
                ExecuteApproach.Result.ERROR_TF,
                f'cannot transform target pose: {exc}')
        except Exception as exc:  # Keep action failures observable to callers.
            self.get_logger().exception(f'execute approach failed: {exc}')
            return self._abort(
                goal_handle,
                result,
                ExecuteApproach.Result.ERROR_EXECUTION_ABORTED,
                str(exc))
        finally:
            with self._active_lock:
                self._goal_active = False

    def _parameter(self, name):
        return self.get_parameter(name).value

    def _validate_goal(self, goal):
        if not goal.target_pose.header.frame_id:
            return 'target_pose.header.frame_id must not be empty'
        if goal.approach_distance < 0.0 or not math.isfinite(
                goal.approach_distance):
            return 'approach_distance must be finite and non-negative'
        if goal.motion not in (ExecuteApproach.Goal.MOTION_PLAN,
                               ExecuteApproach.Goal.MOTION_LINEAR):
            return f'unknown motion {goal.motion}'

        pose = goal.target_pose.pose
        values = (
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w)
        if not all(math.isfinite(value) for value in values):
            return 'target_pose contains a non-finite value'
        norm = math.sqrt(sum(value * value for value in values[3:]))
        if norm < 1e-6:
            return 'target_pose orientation is not a valid quaternion'
        return None

    def _transform_target(self, target_pose):
        target = copy.deepcopy(target_pose)
        orientation = target.pose.orientation
        norm = math.sqrt(
            orientation.x ** 2 + orientation.y ** 2 +
            orientation.z ** 2 + orientation.w ** 2)
        orientation.x /= norm
        orientation.y /= norm
        orientation.z /= norm
        orientation.w /= norm

        planning_frame = self._parameter('planning_frame')
        if target.header.frame_id == planning_frame:
            return target

        stamp = Time.from_msg(target.header.stamp)
        transform = self._tf_buffer.lookup_transform(
            planning_frame,
            target.header.frame_id,
            stamp,
            timeout=Duration(seconds=self._parameter('transform_timeout')))
        return do_transform_pose_stamped(target, transform)

    def _tip_to_link(self, tip_pose):
        offset = self._parameter('tcp_offset')
        if abs(offset) < 1e-9:
            return tip_pose
        link_pose = copy.deepcopy(tip_pose)
        axis = self._tool_z_axis(link_pose.pose.orientation)
        link_pose.pose.position.x -= axis[0] * offset
        link_pose.pose.position.y -= axis[1] * offset
        link_pose.pose.position.z -= axis[2] * offset
        return link_pose

    def _linear(self, goal_handle, result, deadline, target_pose, distance):
        """Straight line from the current pose to target_pose, then along +Z."""
        wait_error = self._wait_for_moveit(goal_handle, deadline)
        if wait_error:
            return self._wait_failure(goal_handle, result, wait_error)
        self._feedback(
            goal_handle,
            ExecuteApproach.Feedback.PHASE_PLANNING_APPROACH,
            0.0,
            'planning linear motion')
        request = self._cartesian_request(target_pose, distance, include_target=True)
        response, wait_error = self._run_future(
            self._cartesian_path.call_async(request), goal_handle, deadline)
        if wait_error:
            return self._wait_failure(goal_handle, result, wait_error)
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return self._moveit_failure(
                goal_handle, result, response.error_code, 'linear planning failed')
        minimum_fraction = self._parameter('minimum_cartesian_fraction')
        if response.fraction < minimum_fraction:
            return self._abort(
                goal_handle,
                result,
                ExecuteApproach.Result.ERROR_CARTESIAN_INCOMPLETE,
                f'linear path fraction {response.fraction:.3f} '
                f'is below required {minimum_fraction:.3f}')
        execute_goal = ExecuteTrajectory.Goal()
        execute_goal.trajectory = response.solution
        self._feedback(
            goal_handle,
            ExecuteApproach.Feedback.PHASE_APPROACHING,
            0.0,
            'executing linear motion')
        execute_result, wait_error = self._run_action(
            self._execute_trajectory, execute_goal, goal_handle, deadline)
        if wait_error:
            return self._wait_failure(goal_handle, result, wait_error)
        if execute_result.error_code.val != MoveItErrorCodes.SUCCESS:
            return self._moveit_failure(
                goal_handle, result, execute_result.error_code,
                'linear execution failed')
        return self._succeed(goal_handle, result)

    def _wait_for_moveit(self, goal_handle, deadline):
        clients = (
            (self._move_group.wait_for_server, '/move_action'),
            (self._cartesian_path.wait_for_service, '/compute_cartesian_path'),
            (self._execute_trajectory.wait_for_server, '/execute_trajectory'),
        )
        server_deadline = min(
            deadline, time.monotonic() + self._parameter('server_timeout'))
        for wait_for_endpoint, name in clients:
            while not wait_for_endpoint(timeout_sec=0.1):
                if goal_handle.is_cancel_requested:
                    return 'cancelled'
                if time.monotonic() >= server_deadline:
                    self.get_logger().error(f'MoveIt endpoint unavailable: {name}')
                    return 'unavailable'
        return None

    def _move_group_goal(self, target_pose):
        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = self._parameter('planning_group')
        request.num_planning_attempts = 3
        request.allowed_planning_time = self._parameter('planning_time')
        request.max_velocity_scaling_factor = self._parameter('velocity_scaling')
        request.max_acceleration_scaling_factor = self._parameter(
            'acceleration_scaling')
        request.start_state.is_diff = True
        request.goal_constraints = [self._pose_constraints(target_pose)]

        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        goal.planning_options.replan_delay = 0.2
        return goal

    def _pose_constraints(self, target_pose):
        tool_link = self._parameter('tool_link')
        position_tolerance = self._parameter('position_tolerance')
        orientation_tolerance = self._parameter('orientation_tolerance')

        position = PositionConstraint()
        position.header = target_pose.header
        position.link_name = tool_link
        region = SolidPrimitive()
        region.type = SolidPrimitive.SPHERE
        region.dimensions = [position_tolerance]
        position.constraint_region.primitives = [region]
        position.constraint_region.primitive_poses = [target_pose.pose]
        position.weight = 1.0

        orientation = OrientationConstraint()
        orientation.header = target_pose.header
        orientation.link_name = tool_link
        orientation.orientation = target_pose.pose.orientation
        orientation.absolute_x_axis_tolerance = orientation_tolerance
        orientation.absolute_y_axis_tolerance = orientation_tolerance
        orientation.absolute_z_axis_tolerance = orientation_tolerance
        orientation.weight = 1.0

        constraints = Constraints()
        constraints.name = 'execute_approach_target'
        constraints.position_constraints = [position]
        constraints.orientation_constraints = [orientation]
        return constraints

    def _cartesian_request(self, target_pose, distance, include_target=False):
        endpoint = copy.deepcopy(target_pose.pose)
        axis = self._tool_z_axis(endpoint.orientation)
        endpoint.position.x += axis[0] * distance
        endpoint.position.y += axis[1] * distance
        endpoint.position.z += axis[2] * distance
        waypoints = [copy.deepcopy(target_pose.pose)] if include_target else []
        if not include_target or distance > 1e-6:
            waypoints.append(endpoint)

        request = GetCartesianPath.Request()
        request.header = target_pose.header
        request.start_state.is_diff = True
        request.group_name = self._parameter('planning_group')
        request.link_name = self._parameter('tool_link')
        request.waypoints = waypoints
        request.max_step = self._parameter('cartesian_step')
        request.jump_threshold = 0.0
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = self._parameter('velocity_scaling')
        request.max_acceleration_scaling_factor = self._parameter(
            'acceleration_scaling')
        return request

    @staticmethod
    def _tool_z_axis(orientation):
        x = orientation.x
        y = orientation.y
        z = orientation.z
        w = orientation.w
        return (
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        )

    def _run_action(
            self, client, action_goal, goal_handle, deadline,
            feedback_callback=None):
        send_future = client.send_goal_async(
            action_goal, feedback_callback=feedback_callback)
        remote_goal, wait_error = self._run_future(
            send_future, goal_handle, deadline)
        if wait_error:
            send_future.add_done_callback(self._cancel_late_goal)
            return None, wait_error
        if not remote_goal.accepted:
            return None, 'rejected'

        result_future = remote_goal.get_result_async()
        wrapped_result, wait_error = self._run_future(
            result_future, goal_handle, deadline, remote_goal)
        if wait_error:
            return None, wait_error
        return wrapped_result.result, None

    @staticmethod
    def _cancel_and_settle(remote_goal, result_future, grace_s=10.0):
        """Cancel and wait for the goal to end before anything new is sent.

        move_group aborts (rcl_action invalid transition EXECUTING -> CANCELED)
        when a new MoveGroup goal arrives while the cancelled one is still
        winding down.
        """
        if remote_goal is None:
            return
        remote_goal.cancel_goal_async()
        deadline = time.monotonic() + grace_s
        while not result_future.done() and time.monotonic() < deadline:
            time.sleep(0.05)

    @staticmethod
    def _cancel_late_goal(send_future):
        try:
            remote_goal = send_future.result()
            if remote_goal is not None and remote_goal.accepted:
                remote_goal.cancel_goal_async()
        except Exception:
            pass

    @staticmethod
    def _run_future(future, goal_handle, deadline, remote_goal=None):
        while not future.done():
            if goal_handle.is_cancel_requested:
                ExecuteApproachNode._cancel_and_settle(remote_goal, future)
                return None, 'cancelled'
            if time.monotonic() >= deadline:
                ExecuteApproachNode._cancel_and_settle(remote_goal, future)
                return None, 'timeout'
            time.sleep(0.05)
        exception = future.exception()
        if exception is not None:
            raise exception
        return future.result(), None

    def _move_group_feedback(self, goal_handle):
        def callback(feedback_message):
            state = feedback_message.feedback.state
            state_lower = state.lower()
            if 'plan' in state_lower:
                phase = ExecuteApproach.Feedback.PHASE_PLANNING
            else:
                phase = ExecuteApproach.Feedback.PHASE_MOVING_TO_STANDOFF
            self._feedback(goal_handle, phase, 0.5, state)
        return callback

    @staticmethod
    def _feedback(goal_handle, phase, progress, status):
        feedback = ExecuteApproach.Feedback()
        feedback.phase = phase
        feedback.progress = float(progress)
        feedback.status = status
        goal_handle.publish_feedback(feedback)

    def _moveit_failure(self, goal_handle, result, moveit_error, prefix):
        code = self._map_moveit_error(moveit_error.val)
        detail = moveit_error.message or str(moveit_error.val)
        return self._abort(
            goal_handle,
            result,
            code,
            f'{prefix}: {detail}',
            moveit_error.val)

    @staticmethod
    def _map_moveit_error(moveit_code):
        if moveit_code in (
            MoveItErrorCodes.INVALID_GROUP_NAME,
            MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS,
            MoveItErrorCodes.GOAL_STATE_INVALID,
            MoveItErrorCodes.UNRECOGNIZED_GOAL_TYPE,
            MoveItErrorCodes.INVALID_LINK_NAME,
        ):
            return ExecuteApproach.Result.ERROR_INVALID_GOAL
        if moveit_code == MoveItErrorCodes.NO_IK_SOLUTION:
            return ExecuteApproach.Result.ERROR_NO_IK
        if moveit_code in (
                MoveItErrorCodes.START_STATE_IN_COLLISION,
                MoveItErrorCodes.GOAL_IN_COLLISION):
            return ExecuteApproach.Result.ERROR_COLLISION
        if moveit_code == MoveItErrorCodes.FRAME_TRANSFORM_FAILURE:
            return ExecuteApproach.Result.ERROR_TF
        if moveit_code == MoveItErrorCodes.TIMED_OUT:
            return ExecuteApproach.Result.ERROR_TIMEOUT
        if moveit_code == MoveItErrorCodes.CONTROL_FAILED:
            return ExecuteApproach.Result.ERROR_CONTROLLER
        if moveit_code in (
                MoveItErrorCodes.COMMUNICATION_FAILURE,
                MoveItErrorCodes.COLLISION_CHECKING_UNAVAILABLE):
            return ExecuteApproach.Result.ERROR_MOVEIT_UNAVAILABLE
        if moveit_code == MoveItErrorCodes.PREEMPTED:
            return ExecuteApproach.Result.ERROR_CANCELLED
        if moveit_code in (
                MoveItErrorCodes.PLANNING_FAILED,
                MoveItErrorCodes.INVALID_MOTION_PLAN,
                MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE,
                MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS,
                MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED):
            return ExecuteApproach.Result.ERROR_PLAN_FAILED
        return ExecuteApproach.Result.ERROR_EXECUTION_ABORTED

    def _wait_failure(self, goal_handle, result, wait_error):
        if wait_error == 'cancelled':
            goal_handle.canceled()
            result.error_code = ExecuteApproach.Result.ERROR_CANCELLED
            result.message = 'approach execution cancelled'
            return result
        if wait_error == 'timeout':
            return self._abort(
                goal_handle,
                result,
                ExecuteApproach.Result.ERROR_TIMEOUT,
                'approach execution timed out')
        if wait_error == 'unavailable':
            return self._abort(
                goal_handle,
                result,
                ExecuteApproach.Result.ERROR_MOVEIT_UNAVAILABLE,
                'required MoveIt endpoint is unavailable')
        return self._abort(
            goal_handle,
            result,
            ExecuteApproach.Result.ERROR_EXECUTION_ABORTED,
            'MoveIt rejected the request')

    def _succeed(self, goal_handle, result):
        self._feedback(
            goal_handle,
            ExecuteApproach.Feedback.PHASE_DONE,
            1.0,
            'approach complete')
        result.success = True
        result.error_code = ExecuteApproach.Result.ERROR_NONE
        result.moveit_error_code = MoveItErrorCodes.SUCCESS
        result.message = 'approach completed successfully'
        goal_handle.succeed()
        return result

    @staticmethod
    def _abort(goal_handle, result, error_code, message, moveit_code=0):
        result.success = False
        result.error_code = error_code
        result.moveit_error_code = moveit_code
        result.message = message
        goal_handle.abort()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = ExecuteApproachNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
