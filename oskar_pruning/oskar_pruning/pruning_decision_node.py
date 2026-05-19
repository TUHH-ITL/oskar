#!/usr/bin/env python3
"""Pruning decision node (Phase 2 — pruning).

Subscribes to pruning targets and cut confirmation.
Ranks flowers by agronomic priority.
Decrements count after each cut.
Stops when current_count <= target_count.
Publishes next target pose and status.
"""

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from oskar_msgs.msg import PruningStatus, PruningTargets


class PruningDecisionNode(Node):
    def __init__(self):
        super().__init__("pruning_decision_node")

        # Declare parameters
        self.declare_parameter("use_sim_time", False)

        # ROS communication
        qos_profile = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.BEST_EFFORT
        )

        # Subscribe to targets and cut confirmation
        self.targets = None
        self.current_target_idx = 0

        self.create_subscription(
            PruningTargets,
            "/pruning/targets",
            self.targets_callback,
            qos_profile,
        )
        self.create_subscription(
            Bool, "/pruning/cut_complete", self.cut_callback, qos_profile
        )

        # Publishers
        self.pub_next_target = self.create_publisher(
            PoseStamped, "/pruning/next_target", qos_profile
        )
        self.pub_status = self.create_publisher(
            PruningStatus, "/pruning/status", qos_profile
        )

        self.get_logger().info("PruningDecisionNode initialized")

    def targets_callback(self, targets_msg):
        """Receive refined pruning targets."""
        self.targets = targets_msg
        self.current_target_idx = 0
        self.publish_next_target()

    def cut_callback(self, msg):
        """Cut confirmed, move to next target."""
        if not self.targets:
            return

        self.targets.current_count = max(0, self.targets.current_count - 1)
        self.current_target_idx += 1

        # Publish status
        status = PruningStatus()
        status.header.stamp = self.get_clock().now().to_msg()
        status.tree_id = self.targets.tree_id
        status.current_count = self.targets.current_count
        status.target_count = self.targets.target_count
        status.done = self.targets.current_count <= self.targets.target_count

        self.pub_status.publish(status)

        if not status.done:
            self.publish_next_target()
        else:
            self.get_logger().info(
                f"Pruning complete for tree {self.targets.tree_id}"
            )

    def publish_next_target(self):
        """Publish the next target flower to prune."""
        if (
            not self.targets
            or self.targets.current_count <= self.targets.target_count
        ):
            return

        if self.current_target_idx >= len(self.targets.ranked_targets):
            self.get_logger().warn("No more targets available")
            return

        target_flower = self.targets.ranked_targets[self.current_target_idx]

        # Create pose message
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position = target_flower.position
        pose.pose.orientation.w = 1.0  # Identity rotation

        self.pub_next_target.publish(pose)

        self.get_logger().debug(
            f"Next target: flower {target_flower.landmark_id} at "
            f"({target_flower.position.x:.3f}, {target_flower.position.y:.3f}, {target_flower.position.z:.3f})",
        )


def main(args=None):
    rclpy.init(args=args)
    node = PruningDecisionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
