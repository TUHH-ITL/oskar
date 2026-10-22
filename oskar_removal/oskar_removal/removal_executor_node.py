#!/usr/bin/env python3
"""Execute a RemovalSequence: per target, stand-off, approach, remove, retract.

Motion goes through ExecuteApproach and the tool through SetToolState, so this
node holds no robot specifics. With use_local_scene, the collision octomap is
rebuilt around each target at the stand-off, with the target and its approach
corridor carved out, before the linear approach.
"""

import time

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from oskar_msgs.action import ExecuteApproach, RemoveTargets
from oskar_msgs.msg import ThinningCandidate, RemovalResult
from oskar_msgs.srv import SetToolState
from oskar_removal.conversions import pose_from, position_of, quaternion_of, stamped


class StepFailed(Exception):
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


APPROACH_FAILURES = {
    ExecuteApproach.Result.ERROR_NO_IK: RemovalResult.FAILURE_UNREACHABLE,
    ExecuteApproach.Result.ERROR_COLLISION: RemovalResult.FAILURE_COLLISION,
    ExecuteApproach.Result.ERROR_CARTESIAN_INCOMPLETE: RemovalResult.FAILURE_COLLISION,
    ExecuteApproach.Result.ERROR_CANCELLED: RemovalResult.FAILURE_ABORTED,
}


class RemovalExecutorNode(Node):
    def __init__(self):
        super().__init__("removal_executor_node")
        self.declare_parameter("use_local_scene", False)
        self.declare_parameter("scene_namespace", "/scene")
        self.declare_parameter("scene_capture_timeout_s", 10.0)
        self.declare_parameter("step_timeout_s", 90.0)

        self._callbacks = ReentrantCallbackGroup()
        self._busy = False
        self._approach = ActionClient(self, ExecuteApproach, "execute_approach",
                                      callback_group=self._callbacks)
        self._tool = self.create_client(SetToolState, "set_tool_state",
                                        callback_group=self._callbacks)
        ns = self.get_parameter("scene_namespace").value.rstrip("/")
        self._scene_target = self.create_publisher(ThinningCandidate, f"{ns}/target", 1)
        self._scene_capture = self.create_client(Trigger, f"{ns}/begin_arrival_capture",
                                                 callback_group=self._callbacks)
        self._scene_build = self.create_client(Trigger, f"{ns}/build",
                                               callback_group=self._callbacks)
        self._arrival_ready = False
        self.create_subscription(Bool, f"{ns}/arrival_ready", self._on_arrival_ready, 1,
                                 callback_group=self._callbacks)

        self._server = ActionServer(
            self, RemoveTargets, "remove_targets", self._execute,
            goal_callback=self._on_goal, cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=self._callbacks)
        self.get_logger().info("removal executor ready")

    def _on_goal(self, _request):
        if self._busy:
            self.get_logger().warn("rejecting goal: a removal sequence is already running")
            return GoalResponse.REJECT
        self._busy = True
        return GoalResponse.ACCEPT

    def _on_arrival_ready(self, msg):
        self._arrival_ready = msg.data

    def _execute(self, goal_handle):
        try:
            return self._run(goal_handle)
        finally:
            self._busy = False

    def _run(self, goal_handle):
        goal = goal_handle.request
        targets = goal.sequence.targets
        result = RemoveTargets.Result()
        feedback = RemoveTargets.Feedback()
        feedback.total = len(targets)

        for index, target in enumerate(targets):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return self._finish(result)
            feedback.current_index = index
            feedback.current_candidate_id = target.candidate.id
            started = self.get_clock().now()
            outcome = RemovalResult()
            outcome.candidate_id = target.candidate.id
            try:
                self._remove_one(goal_handle, goal, target, feedback)
                outcome.success = True
                outcome.failure_reason = RemovalResult.FAILURE_NONE
                outcome.message = "removed" if not goal.dry_run else "reached (dry run)"
            except StepFailed as failure:
                outcome.success = False
                outcome.failure_reason = failure.reason
                outcome.message = str(failure)
                self.get_logger().warn(f"candidate {target.candidate.id}: {failure}")
            outcome.duration = (self.get_clock().now() - started).to_msg()
            result.results.append(outcome)
            if not outcome.success and goal.stop_on_failure:
                break

        goal_handle.succeed()
        return self._finish(result)

    def _finish(self, result):
        result.num_removed = sum(r.success for r in result.results)
        result.num_failed = sum(not r.success for r in result.results)
        result.success = result.num_failed == 0
        self.get_logger().info(f"removal: {result.num_removed} removed, "
                               f"{result.num_failed} failed")
        return result

    def _remove_one(self, goal_handle, goal, target, feedback):
        approach = target.approach_pose
        contact = stamped(pose_from(position_of(target.candidate.pose.pose),
                                    quaternion_of(approach.pose)), approach.header)

        self._status(goal_handle, feedback, "moving to stand-off")
        self._move(approach, ExecuteApproach.Goal.MOTION_PLAN, goal_handle,
                   RemovalResult.FAILURE_PLANNING)

        if self.get_parameter("use_local_scene").value:
            self._status(goal_handle, feedback, "building local scene")
            self._build_scene(target.candidate)

        if not goal.dry_run:
            self._tool_command(SetToolState.Request.CMD_PREPARE)

        self._status(goal_handle, feedback, "approaching")
        self._move(contact, ExecuteApproach.Goal.MOTION_LINEAR, goal_handle,
                   RemovalResult.FAILURE_COLLISION, retract_to=approach)

        if not goal.dry_run:
            self._status(goal_handle, feedback, "removing")
            try:
                self._tool_command(SetToolState.Request.CMD_REMOVE)
            finally:
                self._retract(goal_handle, feedback, approach)
        else:
            self._retract(goal_handle, feedback, approach)

    def _retract(self, goal_handle, feedback, approach):
        self._status(goal_handle, feedback, "retracting")
        self._move(approach, ExecuteApproach.Goal.MOTION_LINEAR, goal_handle,
                   RemovalResult.FAILURE_PLANNING)

    def _move(self, pose, motion, goal_handle, failure, retract_to=None):
        request = ExecuteApproach.Goal()
        request.target_pose = pose
        request.approach_distance = 0.0
        request.motion = motion
        response = self._call_action(request, goal_handle)
        if response is None:
            raise StepFailed(RemovalResult.FAILURE_ABORTED, "motion cancelled or timed out")
        if not response.success:
            if retract_to is not None:
                # A partial linear approach leaves the tool in the canopy.
                self._move(retract_to, ExecuteApproach.Goal.MOTION_LINEAR, goal_handle,
                           RemovalResult.FAILURE_PLANNING)
            raise StepFailed(APPROACH_FAILURES.get(response.error_code, failure),
                             response.message)

    def _call_action(self, request, goal_handle):
        if not self._approach.wait_for_server(timeout_sec=5.0):
            raise StepFailed(RemovalResult.FAILURE_PLANNING, "execute_approach is unavailable")
        sent = self._approach.send_goal_async(request)
        remote = self._wait(sent, goal_handle)
        if remote is None or not remote.accepted:
            return None
        outcome = self._wait(remote.get_result_async(), goal_handle, remote)
        return None if outcome is None else outcome.result

    def _wait(self, future, goal_handle, remote=None):
        deadline = time.monotonic() + self.get_parameter("step_timeout_s").value
        while not future.done():
            if goal_handle.is_cancel_requested or time.monotonic() > deadline:
                if remote is not None:
                    # The next goal must not reach move_group before this one
                    # has wound down; it aborts on the overlap.
                    remote.cancel_goal_async()
                    settle = time.monotonic() + 15.0
                    while not future.done() and time.monotonic() < settle:
                        time.sleep(0.02)
                return None
            time.sleep(0.02)
        return future.result()

    def _tool_command(self, command):
        if not self._tool.wait_for_service(timeout_sec=5.0):
            raise StepFailed(RemovalResult.FAILURE_END_EFFECTOR, "set_tool_state is unavailable")
        request = SetToolState.Request()
        request.command = command
        future = self._tool.call_async(request)
        deadline = time.monotonic() + self.get_parameter("step_timeout_s").value
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        response = future.result() if future.done() else None
        if response is None or not response.success:
            raise StepFailed(RemovalResult.FAILURE_END_EFFECTOR,
                             response.message if response else "tool timed out")

    def _build_scene(self, candidate):
        timeout = self.get_parameter("scene_capture_timeout_s").value
        self._scene_target.publish(candidate)
        self._arrival_ready = False
        self._trigger(self._scene_capture, timeout)
        deadline = time.monotonic() + timeout
        while not self._arrival_ready and time.monotonic() < deadline:
            time.sleep(0.05)
        if not self._arrival_ready:
            raise StepFailed(RemovalResult.FAILURE_PLANNING, "local scene capture timed out")
        self._trigger(self._scene_build, timeout)

    def _trigger(self, client, timeout):
        if not client.wait_for_service(timeout_sec=timeout):
            raise StepFailed(RemovalResult.FAILURE_PLANNING, f"{client.srv_name} is unavailable")
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        response = future.result() if future.done() else None
        if response is None or not response.success:
            raise StepFailed(RemovalResult.FAILURE_PLANNING,
                             f"{client.srv_name}: {response.message if response else 'timeout'}")

    @staticmethod
    def _status(goal_handle, feedback, text):
        feedback.status = text
        goal_handle.publish_feedback(feedback)


def main(args=None):
    rclpy.init(args=args)
    node = RemovalExecutorNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
