#!/usr/bin/env python3
"""Plan how to reach the candidates chosen for removal. Computation only, no motion.

  compute_approach_pose    ComputeApproachPose   tool tip stand-off for one candidate
  plan_removal_sequence    PlanRemovalSequence   order, approach poses, reachability

Reachability asks MoveIt's /compute_ik for a collision-free solution at the
stand-off and at the object. A candidate that fails either is rejected; the
caller keeps it and decides again, so the limit is met with objects the arm can
actually reach.
"""

import time

import numpy as np
import rclpy
import tf2_ros
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose_stamped

from oskar_msgs.action import PlanRemovalSequence
from oskar_msgs.msg import RemovalTarget
from oskar_msgs.srv import ComputeApproachPose
from oskar_removal.approach import approach_pose, matrix_from_quaternion, tip_to_link
from oskar_removal.conversions import pose_from, position_of, quaternion_of, stamped
from oskar_removal.ordering import order_nearest_neighbour


class RemovalPlannerNode(Node):
    def __init__(self):
        super().__init__("removal_planner_node")
        self.declare_parameter("roll_reference", [0.0, 0.0, 1.0])
        self.declare_parameter("default_standoff_m", 0.10)
        # Tool tip in front of tool_link along its +Z: the Robotiq 2F-85 finger
        # pad centre in tipard_ur20_combined.urdf.
        self.declare_parameter("tcp_offset_m", 0.130)
        self.declare_parameter("tool_link", "tool0")
        self.declare_parameter("planning_group", "ur_manipulator")
        self.declare_parameter("planning_frame", "base_link")
        self.declare_parameter("ik_timeout_s", 0.1)
        # KDL restarts from random seeds, so one failed call is not proof the
        # pose is out of reach; a rejected candidate stays on the tree for good.
        self.declare_parameter("ik_attempts", 3)
        self.declare_parameter("service_timeout_s", 5.0)

        self._callbacks = ReentrantCallbackGroup()
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self, spin_thread=False)
        self._ik = self.create_client(GetPositionIK, "/compute_ik",
                                      callback_group=self._callbacks)

        self.create_service(ComputeApproachPose, "compute_approach_pose",
                            self._compute_approach_pose, callback_group=self._callbacks)
        self._sequence_server = ActionServer(
            self, PlanRemovalSequence, "plan_removal_sequence", self._plan_sequence,
            cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=self._callbacks)
        self.get_logger().info("removal planner ready")

    def _param(self, name):
        return self.get_parameter(name).value

    # --- ComputeApproachPose --------------------------------------------------

    def _compute_approach_pose(self, request, response):
        c = request.candidate
        if not c.pose.header.frame_id:
            response.success = False
            response.message = "candidate pose has an empty frame_id"
            return response
        standoff = request.standoff if request.standoff > 0 else self._param("default_standoff_m")
        tip, q = self._approach(c, standoff)
        response.pose = stamped(pose_from(tip, q), c.pose.header)
        response.success = True
        response.message = "approach pose computed"
        return response

    def _approach(self, candidate, standoff):
        axis = matrix_from_quaternion(quaternion_of(candidate.pose.pose))[:, 2]
        return approach_pose(position_of(candidate.pose.pose), axis, standoff,
                             self._param("roll_reference"))

    # --- PlanRemovalSequence --------------------------------------------------

    def _plan_sequence(self, goal_handle):
        goal = goal_handle.request
        candidates = list(goal.candidates.candidates)
        standoff = goal.standoff if goal.standoff > 0 else self._param("default_standoff_m")
        result = PlanRemovalSequence.Result()
        result.sequence.header = goal.candidates.header
        result.sequence.tree_id = goal.candidates.tree_id

        start = None
        if goal.start.header.frame_id:
            start = self._start_in(goal.start, goal.candidates.header.frame_id)
        positions = np.array([position_of(c.pose.pose) for c in candidates]).reshape(-1, 3)
        order = order_nearest_neighbour(positions, start)

        check = goal.check_reachability
        if check and not self._ik.wait_for_service(timeout_sec=self._param("service_timeout_s")):
            result.success = False
            result.message = "MoveIt /compute_ik is unavailable"
            goal_handle.abort()
            return result

        feedback = PlanRemovalSequence.Feedback()
        feedback.total = len(order)
        for i in order:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.message = "cancelled"
                return result
            c = candidates[i]
            tip, q = self._approach(c, standoff)
            target = RemovalTarget()
            target.candidate = c
            target.approach_pose = stamped(pose_from(tip, q), c.pose.header)
            target.approach_distance = float(standoff)
            contact = stamped(pose_from(position_of(c.pose.pose), q), c.pose.header)
            if check and not (self._reachable(target.approach_pose)
                              and self._reachable(contact)):
                result.rejected_ids.append(c.id)
                feedback.rejected += 1
            else:
                result.sequence.targets.append(target)
                feedback.planned += 1
            goal_handle.publish_feedback(feedback)

        result.success = True
        result.message = (f"{len(result.sequence.targets)} planned, "
                          f"{len(result.rejected_ids)} unreachable")
        self.get_logger().info(
            f"tree {goal.candidates.tree_id}: removal sequence, {result.message}")
        goal_handle.succeed()
        return result

    def _start_in(self, start, frame):
        try:
            if start.header.frame_id != frame:
                start = self._transform(start, frame)
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(f"start pose not transformable, ordering from the top: {exc}")
            return None
        return position_of(start.pose)

    def _transform(self, pose_stamped, frame):
        transform = self._tf_buffer.lookup_transform(
            frame, pose_stamped.header.frame_id, Time(), timeout=Duration(seconds=1.0))
        return do_transform_pose_stamped(pose_stamped, transform)

    def _reachable(self, tip_pose):
        """Collision-free IK exists for the tool tip at this pose."""
        try:
            in_planning = self._transform(tip_pose, self._param("planning_frame"))
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(f"reachability: {exc}")
            return False
        link, q = tip_to_link(position_of(in_planning.pose), quaternion_of(in_planning.pose),
                              self._param("tcp_offset_m"))
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = self._param("planning_group")
        ik.ik_link_name = self._param("tool_link")
        ik.avoid_collisions = True
        ik.robot_state.is_diff = True
        ik.pose_stamped = stamped(pose_from(link, q), in_planning.header)
        ik.timeout = Duration(seconds=self._param("ik_timeout_s")).to_msg()
        for _ in range(max(1, int(self._param("ik_attempts")))):
            future = self._ik.call_async(request)
            deadline = time.monotonic() + self._param("service_timeout_s")
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.005)
            if (future.done() and future.result() is not None
                    and future.result().error_code.val == MoveItErrorCodes.SUCCESS):
                return True
        return False


def main(args=None):
    rclpy.init(args=args)
    node = RemovalPlannerNode()
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
