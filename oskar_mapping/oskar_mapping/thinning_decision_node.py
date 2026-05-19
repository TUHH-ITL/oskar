#!/usr/bin/env python3
"""
Thinning decision node.

Analyzes per-tree flower counts.
Decides whether thinning is needed.
Computes rough removal count and priority scores.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from oskar_msgs.msg import PerTreeFlowers, ThinningPlan


class ThinningDecisionNode(Node):
    def __init__(self):
        super().__init__('thinning_decision_node')
        
        # Declare parameters
        self.declare_parameter('target_flowers_per_tree', 20)
        self.declare_parameter('agronomic_min', 15)
        self.declare_parameter('agronomic_max', 25)
        
        # Get parameters
        self.target = self.get_parameter('target_flowers_per_tree').value
        self.agronomic_min = self.get_parameter('agronomic_min').value
        self.agronomic_max = self.get_parameter('agronomic_max').value
        
        self.get_logger().info(f"Target flowers: {self.target}")
        self.get_logger().info(f"Agronomic range: [{self.agronomic_min}, {self.agronomic_max}]")
        
        # ROS communication
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(PerTreeFlowers, '/orchard/per_tree_flowers',
                                          self.flowers_callback, qos_profile)
        self.pub = self.create_publisher(ThinningPlan, '/orchard/thinning_plan', qos_profile)
        
        self.get_logger().info("ThinningDecisionNode initialized")
    
    def flowers_callback(self, flowers_msg):
        """Make thinning decisions per tree."""
        try:
            # Create output message
            plan_msg = ThinningPlan()
            plan_msg.header.stamp = flowers_msg.header.stamp
            plan_msg.header.frame_id = flowers_msg.header.frame_id
            
            # Process each tree
            for i, tree_id in enumerate(flowers_msg.tree_ids):
                confirmed = flowers_msg.confirmed_counts[i]
                inferred = flowers_msg.inferred_counts[i]
                total = confirmed + inferred
                
                # Decision logic
                needs_thinning = total > self.agronomic_max
                rough_removal = max(0, total - self.target)
                priority = (total - self.target) / self.target if self.target > 0 else 0.0
                
                plan_msg.tree_ids.append(tree_id)
                plan_msg.needs_thinning.append(needs_thinning)
                plan_msg.rough_removal_count.append(rough_removal)
                plan_msg.priority_score.append(max(0.0, priority))
                
                self.get_logger().debug(
                    f"Tree {tree_id}: {total} flowers (conf={confirmed}, inf={inferred}), "
                    f"needs_thinning={needs_thinning}, remove={rough_removal}"
                )
            
            self.pub.publish(plan_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error in flowers_callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ThinningDecisionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
