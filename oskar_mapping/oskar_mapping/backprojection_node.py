#!/usr/bin/env python3
"""
Backprojection node.

Transforms 3D flower detections from camera frame to map frame.
Uses TF2 to look up transform at detection timestamp.
Publishes world-frame observations as FlowerLandmarks.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped

from oskar_msgs.msg import FlowerDetections3D, FlowerLandmark, FlowerLandmarks


class BackprojectionNode(Node):
    def __init__(self):
        super().__init__('backprojection_node')
        
        # Declare parameters
        self.declare_parameter('camera_frame', 'camera_left_optical')
        self.declare_parameter('map_frame', 'map')
        
        # Get parameters
        self.camera_frame = self.get_parameter('camera_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        
        self.get_logger().info(f"Camera frame: {self.camera_frame}")
        self.get_logger().info(f"Map frame: {self.map_frame}")
        
        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # ROS communication
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(FlowerDetections3D, '/flowers/detections_3d',
                                          self.detections_callback, qos_profile)
        self.pub = self.create_publisher(FlowerLandmarks, '/flowers/world_obs', qos_profile)
        
        self.landmark_id_counter = 0
        self.get_logger().info("BackprojectionNode initialized")
    
    def detections_callback(self, detections_msg):
        """Transform detections to map frame."""
        try:
            # Create output message
            landmarks_msg = FlowerLandmarks()
            landmarks_msg.header.stamp = detections_msg.header.stamp
            landmarks_msg.frame_id = self.map_frame
            
            # Try to get transform
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame, 
                    self.camera_frame,
                    detections_msg.header.stamp,
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
            except tf2_ros.TransformException as e:
                self.get_logger().warn(f"TF lookup failed: {e}")
                return
            
            # Transform each detection
            for detection in detections_msg.detections:
                # Create PointStamped in camera frame
                point_cam = PointStamped()
                point_cam.header.stamp = detections_msg.header.stamp
                point_cam.header.frame_id = self.camera_frame
                point_cam.point = detection.position
                
                # Transform to map frame
                try:
                    point_map = do_transform_point(point_cam, transform)
                except Exception as e:
                    self.get_logger().warn(f"Transform failed for detection {detection.instance_id}: {e}")
                    continue
                
                # Create landmark
                landmark = FlowerLandmark()
                landmark.landmark_id = self.landmark_id_counter
                self.landmark_id_counter += 1
                landmark.position = point_map.point
                landmark.observation_count = 1
                landmark.mean_confidence = detection.confidence
                landmark.is_inferred = False
                landmark.tree_id = 0  # Will be assigned later
                
                landmarks_msg.landmarks.append(landmark)
            
            self.get_logger().debug(f"Backprojected {len(landmarks_msg.landmarks)} landmarks")
            self.pub.publish(landmarks_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error in detections_callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = BackprojectionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
