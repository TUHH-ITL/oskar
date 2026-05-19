#!/usr/bin/env python3
"""
Multiview fusion node.

Maintains rolling buffer of world-frame flower observations.
Flushes old observations based on robot position or age.
Clusters using DBSCAN.
Publishes merged landmarks.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np
import tf2_ros
from sklearn.cluster import DBSCAN
from collections import deque

from oskar_msgs.msg import FlowerLandmarks, FlowerLandmark


class MultiviewFusionNode(Node):
    def __init__(self):
        super().__init__('multiview_fusion_node')
        
        # Declare parameters
        self.declare_parameter('eps', 0.08)
        self.declare_parameter('min_samples', 3)
        self.declare_parameter('window_m', 4.0)
        self.declare_parameter('max_age_s', 30.0)
        
        # Get parameters
        self.eps = self.get_parameter('eps').value
        self.min_samples = self.get_parameter('min_samples').value
        self.window_m = self.get_parameter('window_m').value
        self.max_age_s = self.get_parameter('max_age_s').value
        
        self.get_logger().info(f"DBSCAN eps: {self.eps} m")
        self.get_logger().info(f"Window: {self.window_m} m")
        self.get_logger().info(f"Max age: {self.max_age_s} s")
        
        # TF2 for robot position
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Rolling buffer of observations
        self.observations = deque()  # List of (landmark, timestamp, position_x)
        self.landmark_id_counter = 0
        self.last_flushed_x = 0.0
        
        # ROS communication
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(FlowerLandmarks, '/flowers/world_obs',
                                          self.obs_callback, qos_profile)
        self.pub = self.create_publisher(FlowerLandmarks, '/flowers/landmarks', qos_profile)
        
        self.get_logger().info("MultiviewFusionNode initialized")
    
    def get_robot_position_x(self, timestamp):
        """Get robot's X position at given timestamp."""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', timestamp,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            return transform.transform.translation.x
        except:
            return None
    
    def flush_buffer(self, current_time, current_x):
        """Remove old observations from buffer."""
        current_time_s = current_time.sec + current_time.nanosec / 1e9
        
        # Flush by robot position (primary)
        if current_x is not None:
            distance_traveled = abs(current_x - self.last_flushed_x)
            if distance_traveled > self.window_m:
                self.observations.clear()
                self.last_flushed_x = current_x
                self.get_logger().debug(f"Flushed buffer: robot moved {distance_traveled:.2f} m")
        
        # Flush by age (fallback for stationary testing)
        new_buffer = deque()
        for landmark, ts, pos_x in self.observations:
            ts_s = ts.sec + ts.nanosec / 1e9
            age = current_time_s - ts_s
            if age < self.max_age_s:
                new_buffer.append((landmark, ts, pos_x))
        
        if len(self.observations) != len(new_buffer):
            self.get_logger().debug(f"Flushed {len(self.observations) - len(new_buffer)} old observations")
        self.observations = new_buffer
    
    def obs_callback(self, landmarks_msg):
        """Process incoming observations."""
        try:
            # Flush old observations
            current_x = self.get_robot_position_x(landmarks_msg.header.stamp)
            self.flush_buffer(landmarks_msg.header.stamp, current_x)
            
            # Add new observations to buffer
            for landmark in landmarks_msg.landmarks:
                self.observations.append((landmark, landmarks_msg.header.stamp, current_x or 0.0))
            
            # Cluster observations in buffer
            if len(self.observations) < self.min_samples:
                self.get_logger().debug(f"Buffer has {len(self.observations)} observations, need {self.min_samples}")
                return
            
            # Extract positions
            positions = np.array([[lm.position.x, lm.position.y, lm.position.z] 
                                 for lm, _, _ in self.observations])
            
            # DBSCAN clustering
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(positions)
            labels = clustering.labels_
            
            # Group observations by cluster
            clusters = {}
            for idx, label in enumerate(labels):
                if label == -1:  # Noise
                    continue
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(idx)
            
            # Create merged landmarks
            output_msg = FlowerLandmarks()
            output_msg.header.stamp = landmarks_msg.header.stamp
            output_msg.header.frame_id = landmarks_msg.frame_id
            
            for cluster_id, indices in clusters.items():
                if len(indices) == 1:  # Discard singletons
                    continue
                
                # Compute confidence-weighted centroid
                cluster_landmarks = [self.observations[i][0] for i in indices]
                confidences = np.array([lm.mean_confidence for lm in cluster_landmarks])
                cluster_positions = np.array([[lm.position.x, lm.position.y, lm.position.z]
                                              for lm in cluster_landmarks])

                weights = confidences / confidences.sum()
                centroid = (cluster_positions * weights[:, np.newaxis]).sum(axis=0)
                mean_conf = confidences.mean()
                
                # Create merged landmark
                merged = FlowerLandmark()
                merged.landmark_id = self.landmark_id_counter
                self.landmark_id_counter += 1
                merged.position.x = float(centroid[0])
                merged.position.y = float(centroid[1])
                merged.position.z = float(centroid[2])
                merged.observation_count = len(indices)
                merged.mean_confidence = float(mean_conf)
                merged.is_inferred = False
                merged.tree_id = 0
                
                output_msg.landmarks.append(merged)
            
            self.get_logger().debug(f"Fused into {len(output_msg.landmarks)} landmarks from {len(clusters)} clusters")
            self.pub.publish(output_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error in obs_callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MultiviewFusionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
