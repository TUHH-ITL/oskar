#!/usr/bin/env python3
"""
Depth fusion node.

Synchronizes depth map and flower masks.
Erodes masks, samples median depth, computes depth_std_ratio.
Publishes 3D flower detections (camera frame).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from message_filters import ApproximateTimeSynchronizer, Subscriber
import cv2
import numpy as np
from cv_bridge import CvBridge

from oskar_msgs.msg import StereoPair, FlowerMasks, FlowerDetection3D, FlowerDetections3D
from sensor_msgs.msg import Image, CameraInfo


class DepthFusionNode(Node):
    def __init__(self):
        super().__init__('depth_fusion_node')
        
        # Declare parameters
        self.declare_parameter('left_info_topic', '/camera/left/camera_info')
        self.declare_parameter('erosion_radius_px', 3)
        self.declare_parameter('depth_std_threshold', 0.15)
        self.declare_parameter('min_valid_pixels', 5)
        
        # Get parameters
        self.left_info_topic = self.get_parameter('left_info_topic').value
        self.erosion_radius_px = self.get_parameter('erosion_radius_px').value
        self.depth_std_threshold = self.get_parameter('depth_std_threshold').value
        self.min_valid_pixels = self.get_parameter('min_valid_pixels').value
        
        self.get_logger().info(f"Erosion radius: {self.erosion_radius_px} px")
        self.get_logger().info(f"Min valid pixels: {self.min_valid_pixels}")
        
        # ROS communication
        # RELIABLE: depth (~6 MB) and masks (tens of MB) are large; BEST_EFFORT drops
        # the fragmented messages on localhost, so the time-sync never pairs them.
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # Subscribe to depth and masks
        depth_sub = Subscriber(self, Image, '/stereo/depth', qos_profile=qos_profile)
        masks_sub = Subscriber(self, FlowerMasks, '/flowers/masks', qos_profile=qos_profile)
        
        # Synchronizer
        self.ts = ApproximateTimeSynchronizer(
            [depth_sub, masks_sub],
            queue_size=10,
            slop=0.05
        )
        self.ts.registerCallback(self.fusion_callback)
        
        # Subscribe to CameraInfo separately (not synchronized)
        self.left_info = None
        self.create_subscription(CameraInfo, self.left_info_topic, 
                               self.info_callback, qos_profile)
        
        # Publisher
        self.pub = self.create_publisher(FlowerDetections3D, '/flowers/detections_3d', qos_profile)
        
        self.bridge = CvBridge()
        self.get_logger().info("DepthFusionNode initialized")
    
    def info_callback(self, msg):
        """Store camera info."""
        self.left_info = msg
    
    def fusion_callback(self, depth_msg, masks_msg):
        """Fuse depth and flowers masks."""
        try:
            if self.left_info is None:
                self.get_logger().warn("CameraInfo not yet available")
                return
            
            # Convert depth to OpenCV
            depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            
            # Extract camera intrinsics
            K = np.array(self.left_info.k).reshape(3, 3)
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            
            # Create output message
            detections_msg = FlowerDetections3D()
            detections_msg.header.stamp = depth_msg.header.stamp
            detections_msg.header.frame_id = depth_msg.header.frame_id
            
            # Create erosion kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (2 * self.erosion_radius_px + 1,
                                               2 * self.erosion_radius_px + 1))
            
            # Process each flower mask
            for flower_mask in masks_msg.flowers:
                try:
                    # Convert mask to OpenCV
                    mask_cv = self.bridge.imgmsg_to_cv2(flower_mask.mask, desired_encoding="mono8")
                    mask_binary = (mask_cv > 128).astype(np.uint8)
                    
                    # Erode mask
                    mask_eroded = cv2.morphologyEx(mask_binary, cv2.MORPH_ERODE, kernel)
                    
                    # Sample depth values
                    depth_samples = depth_cv[mask_eroded > 0]
                    valid_depths = depth_samples[~np.isnan(depth_samples)]
                    
                    if len(valid_depths) < self.min_valid_pixels:
                        continue
                    
                    # Compute median depth
                    median_depth = float(np.median(valid_depths))
                    depth_std = float(np.std(valid_depths))
                    depth_std_ratio = depth_std / median_depth if median_depth > 0 else np.inf
                    low_confidence = depth_std_ratio > self.depth_std_threshold
                    
                    # Backproject centroid to 3D
                    u = flower_mask.centroid_u
                    v = flower_mask.centroid_v
                    
                    x_cam = (u - cx) / fx * median_depth
                    y_cam = (v - cy) / fy * median_depth
                    z_cam = median_depth
                    
                    # Create detection
                    detection = FlowerDetection3D()
                    detection.instance_id = flower_mask.instance_id
                    detection.confidence = flower_mask.confidence
                    detection.low_confidence = low_confidence
                    detection.depth_std_ratio = depth_std_ratio
                    detection.position.x = x_cam
                    detection.position.y = y_cam
                    detection.position.z = z_cam
                    
                    detections_msg.detections.append(detection)
                    
                except Exception as e:
                    self.get_logger().warn(f"Error processing flower {flower_mask.instance_id}: {e}")
                    continue
            
            self.get_logger().debug(f"Fused {len(detections_msg.detections)} 3D detections")
            self.pub.publish(detections_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error in fusion_callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DepthFusionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
