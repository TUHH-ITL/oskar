#!/usr/bin/env python3
"""
Biology sanity check node.

Applies apple flower biology priors to fused landmarks.
Infers missing corymb members.
Merges double-detections.
Promotes singleton flowers to full coymbs.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np
from sklearn.cluster import DBSCAN

from oskar_msgs.msg import FlowerLandmarks, FlowerLandmark


class BioSanityNode(Node):
    def __init__(self):
        super().__init__('bio_sanity_node')
        
        # Declare parameters
        self.declare_parameter('expected_flowers_per_corymb', 5)
        self.declare_parameter('intra_flower_spacing_m', 0.03)
        self.declare_parameter('merge_distance_m', 0.02)
        self.declare_parameter('corymb_radius_m', 0.06)
        
        # Get parameters
        self.expected_per_corymb = self.get_parameter('expected_flowers_per_corymb').value
        self.intra_spacing = self.get_parameter('intra_flower_spacing_m').value
        self.merge_distance = self.get_parameter('merge_distance_m').value
        self.corymb_radius = self.get_parameter('corymb_radius_m').value
        
        self.get_logger().info(f"Expected per corymb: {self.expected_per_corymb}")
        self.get_logger().info(f"Corymb radius: {self.corymb_radius} m")
        
        # ROS communication
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(FlowerLandmarks, '/flowers/landmarks',
                                          self.landmarks_callback, qos_profile)
        self.pub = self.create_publisher(FlowerLandmarks, '/flowers/landmarks_bio', qos_profile)
        
        self.landmark_id_counter = 0
        self.get_logger().info("BioSanityNode initialized")
    
    def landmarks_callback(self, landmarks_msg):
        """Apply biology priors."""
        try:
            landmarks = list(landmarks_msg.landmarks)
            
            # Step 1: Merge double-detections (within merge_distance)
            landmarks = self.merge_detections(landmarks)
            
            # Step 2: Group into corymb candidates
            corymb_groups = self.group_into_corymbs(landmarks)
            
            # Step 3: Infer missing flowers and promote singletons
            output_landmarks = []
            for group in corymb_groups:
                inferred_group = self.infer_corymb(group, landmarks_msg.header.stamp)
                output_landmarks.extend(inferred_group)
            
            # Create output message
            output_msg = FlowerLandmarks()
            output_msg.header.stamp = landmarks_msg.header.stamp
            output_msg.header.frame_id = landmarks_msg.frame_id
            output_msg.landmarks = output_landmarks
            
            self.get_logger().debug(f"Bio sanity: {len(output_landmarks)} landmarks after inference")
            self.pub.publish(output_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error in landmarks_callback: {e}")
    
    def merge_detections(self, landmarks):
        """Merge landmarks within merge_distance."""
        if len(landmarks) < 2:
            return landmarks
        
        positions = np.array([[lm.position.x, lm.position.y, lm.position.z] for lm in landmarks])
        
        # Cluster close landmarks
        clustering = DBSCAN(eps=self.merge_distance, min_samples=1).fit(positions)
        labels = clustering.labels_
        
        merged = {}
        for idx, label in enumerate(labels):
            if label not in merged:
                merged[label] = []
            merged[label].append(landmarks[idx])
        
        result = []
        for cluster_landmarks in merged.values():
            if len(cluster_landmarks) == 1:
                result.append(cluster_landmarks[0])
            else:
                # Merge: keep highest confidence
                best = max(cluster_landmarks, key=lambda lm: lm.mean_confidence)
                result.append(best)
        
        return result
    
    def group_into_corymbs(self, landmarks):
        """Group landmarks into spatial clusters (corymb candidates)."""
        if len(landmarks) == 0:
            return []
        
        positions = np.array([[lm.position.x, lm.position.y, lm.position.z] for lm in landmarks])
        
        # Cluster within corymb_radius (rough clustering)
        clustering = DBSCAN(eps=self.corymb_radius, min_samples=1).fit(positions)
        labels = clustering.labels_
        
        groups = {}
        for idx, label in enumerate(labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(landmarks[idx])
        
        return list(groups.values())
    
    def infer_corymb(self, group, timestamp):
        """
        Infer missing flowers in a group.
        If group has fewer than expected_per_corymb, infer missing ones.
        If singleton, promote to full corymb.
        """
        output = []
        
        # Pass through confirmed flowers
        for landmark in group:
            output.append(landmark)
        
        # Decide how many to infer
        confirmed_count = len(group)
        
        if confirmed_count == 0:
            return output
        elif confirmed_count == 1:
            # Singleton: promote to full corymb
            to_infer = self.expected_per_corymb - 1
        elif confirmed_count < self.expected_per_corymb:
            # Sparse: infer missing
            to_infer = self.expected_per_corymb - confirmed_count
        else:
            # Full or overfull: no inference
            return output
        
        # Compute group centroid
        positions = np.array([[lm.position.x, lm.position.y, lm.position.z] for lm in group])
        centroid = positions.mean(axis=0)
        
        # Generate inferred positions around centroid
        for i in range(to_infer):
            # Angular offset for lateral blooms
            angle = 2 * np.pi * i / to_infer
            offset = self.intra_spacing * 0.5  # Radial offset
            dx = offset * np.cos(angle)
            dy = offset * np.sin(angle)
            
            inferred = FlowerLandmark()
            inferred.landmark_id = self.landmark_id_counter
            self.landmark_id_counter += 1
            inferred.position.x = float(centroid[0] + dx)
            inferred.position.y = float(centroid[1] + dy)
            inferred.position.z = float(centroid[2])
            inferred.observation_count = 0
            inferred.mean_confidence = 0.3  # Low confidence for inferred
            inferred.is_inferred = True
            inferred.tree_id = 0
            
            output.append(inferred)
        
        return output


def main(args=None):
    rclpy.init(args=args)
    node = BioSanityNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
