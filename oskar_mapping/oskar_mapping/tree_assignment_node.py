#!/usr/bin/env python3
"""
Tree assignment node.

Loads tree positions from YAML.
Assigns landmarks to nearest tree within max_radius.
Publishes PerTreeFlowers message.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import yaml
from pathlib import Path
import numpy as np

from oskar_msgs.msg import FlowerLandmarks, PerTreeFlowers


class TreeAssignmentNode(Node):
    def __init__(self):
        super().__init__('tree_assignment_node')
        
        # Declare parameters
        self.declare_parameter('tree_map_file', 'config/tree_map.yaml')
        self.declare_parameter('max_radius_m', 0.8)
        
        # Get parameters
        tree_map_file = self.get_parameter('tree_map_file').value
        self.max_radius = self.get_parameter('max_radius_m').value
        
        self.get_logger().info(f"Tree map file: {tree_map_file}")
        self.get_logger().info(f"Max assignment radius: {self.max_radius} m")
        
        # Load tree positions
        self.trees = {}  # tree_id -> (x, y)
        self.load_tree_map(tree_map_file)
        
        # ROS communication
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(FlowerLandmarks, '/flowers/landmarks_bio',
                                          self.landmarks_callback, qos_profile)
        self.pub = self.create_publisher(PerTreeFlowers, '/orchard/per_tree_flowers', qos_profile)
        
        self.get_logger().info("TreeAssignmentNode initialized")
    
    def load_tree_map(self, tree_map_file):
        """Load tree positions from YAML."""
        try:
            # Resolve relative paths from package
            if not Path(tree_map_file).is_absolute():
                # Try relative to current working directory
                tree_map_path = Path(tree_map_file)
                if not tree_map_path.exists():
                    # Try relative to ROS package
                    import ament_index_python
                    pkg_path = ament_index_python.get_package_share_directory('oskar_mapping')
                    tree_map_path = Path(pkg_path) / tree_map_file
            else:
                tree_map_path = Path(tree_map_file)
            
            if not tree_map_path.exists():
                self.get_logger().warn(f"Tree map not found: {tree_map_path}, using empty map")
                return
            
            with open(tree_map_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if 'trees' not in data:
                self.get_logger().warn("No 'trees' key in tree map")
                return
            
            for tree_id, pos in data['trees'].items():
                self.trees[int(tree_id)] = (float(pos['x']), float(pos['y']))
            
            self.get_logger().info(f"Loaded {len(self.trees)} trees")
        except Exception as e:
            self.get_logger().error(f"Error loading tree map: {e}")
    
    def landmarks_callback(self, landmarks_msg):
        """Assign landmarks to trees."""
        try:
            if not self.trees:
                self.get_logger().warn("No trees loaded, skipping assignment")
                return
            
            # Initialize per-tree counts
            per_tree = {tree_id: {'confirmed': 0, 'inferred': 0, 'landmarks': []} 
                       for tree_id in self.trees}
            
            # Assign each landmark
            for landmark in landmarks_msg.landmarks:
                closest_tree_id = None
                closest_distance = self.max_radius
                
                # Find nearest tree
                for tree_id, (tree_x, tree_y) in self.trees.items():
                    dist = np.sqrt((landmark.position.x - tree_x)**2 + 
                                 (landmark.position.y - tree_y)**2)
                    if dist < closest_distance:
                        closest_distance = dist
                        closest_tree_id = tree_id
                
                if closest_tree_id is not None:
                    landmark.tree_id = closest_tree_id
                    per_tree[closest_tree_id]['landmarks'].append(landmark)
                    if landmark.is_inferred:
                        per_tree[closest_tree_id]['inferred'] += 1
                    else:
                        per_tree[closest_tree_id]['confirmed'] += 1
            
            # Create output message
            output_msg = PerTreeFlowers()
            output_msg.header.stamp = landmarks_msg.header.stamp
            output_msg.header.frame_id = landmarks_msg.frame_id
            
            from oskar_msgs.msg import FlowerLandmarks as FlowerLandmarksMsg
            
            for tree_id in sorted(per_tree.keys()):
                output_msg.tree_ids.append(tree_id)
                output_msg.confirmed_counts.append(per_tree[tree_id]['confirmed'])
                output_msg.inferred_counts.append(per_tree[tree_id]['inferred'])
                
                # Create per-tree landmarks message
                tree_landmarks = FlowerLandmarksMsg()
                tree_landmarks.header = output_msg.header
                tree_landmarks.frame_id = landmarks_msg.frame_id
                tree_landmarks.landmarks = per_tree[tree_id]['landmarks']
                output_msg.per_tree_landmarks.append(tree_landmarks)
            
            self.get_logger().debug(f"Assigned {len(landmarks_msg.landmarks)} landmarks to {len(per_tree)} trees")
            self.pub.publish(output_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error in landmarks_callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TreeAssignmentNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
