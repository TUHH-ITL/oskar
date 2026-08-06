#!/usr/bin/env python3
"""
Apple flower instance segmentation node using Detectron2 Mask R-CNN.

Subscribes to StereoPair (uses left image only).
Publishes FlowerMasks with instance segmentation results.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import cv2
import numpy as np
from cv_bridge import CvBridge
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from pathlib import Path
import torch

from oskar_msgs.msg import StereoPair, FlowerMask, FlowerMasks


class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')
        
        # Declare parameters
        self.declare_parameter('model_path', 
                             '/home/workstation/oskar/synthetic_apple_flowers/results/exp3_real_real/train/model_final.pth')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('device', 'cuda')
        
        # Get parameters
        model_path = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        device = self.get_parameter('device').value
        
        self.get_logger().info(f"Loading model from: {model_path}")
        self.get_logger().info(f"Confidence threshold: {self.confidence_threshold}")
        self.get_logger().info(f"Device: {device}")
        
        # Load Detectron2 model
        self.predictor = self.load_model(model_path, device)
        
        # ROS communication
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(StereoPair, '/stereo/sync_pair',
                                          self.image_callback, qos_profile)
        self.pub = self.create_publisher(FlowerMasks, '/flowers/masks', qos_profile)
        
        self.bridge = CvBridge()
        self.get_logger().info("SegmentationNode initialized")
    
    def load_model(self, model_path, device):
        """Load Detectron2 Mask R-CNN model."""
        try:
            model_path = Path(model_path)
            config_path = model_path.parent / "detectron2_config.yaml"
            
            cfg = get_cfg()
            cfg.merge_from_file(str(config_path))
            cfg.MODEL.WEIGHTS = str(model_path)
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
            cfg.MODEL.DEVICE = device
            
            predictor = DefaultPredictor(cfg)
            self.get_logger().info("Model loaded successfully")
            return predictor
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise
    
    def image_callback(self, stereo_pair_msg):
        """Process left image and generate flower masks."""
        try:
            # Convert ROS image to OpenCV
            left_cv = self.bridge.imgmsg_to_cv2(stereo_pair_msg.left_image, desired_encoding="bgr8")
            
            # Run inference
            outputs = self.predictor(left_cv)
            
            # Extract predictions
            instances = outputs["instances"]
            masks = instances.pred_masks.cpu().numpy()  # (N, H, W)
            scores = instances.scores.cpu().numpy()     # (N,)
            
            # Create FlowerMasks message
            flower_masks_msg = FlowerMasks()
            flower_masks_msg.header.stamp = stereo_pair_msg.header.stamp
            flower_masks_msg.header.frame_id = stereo_pair_msg.header.frame_id
            
            # Process each detected flower
            for instance_id, (mask, score) in enumerate(zip(masks, scores)):
                if score < self.confidence_threshold:
                    continue
                
                # Compute centroid
                y_coords, x_coords = np.where(mask)
                if len(x_coords) == 0:
                    continue
                centroid_u = float(np.mean(x_coords))
                centroid_v = float(np.mean(y_coords))
                
                # Create FlowerMask message
                flower_mask = FlowerMask()
                flower_mask.instance_id = instance_id
                flower_mask.confidence = float(score)
                flower_mask.centroid_u = centroid_u
                flower_mask.centroid_v = centroid_v
                
                # Convert mask to ROS Image message (uint8, 0=background, 255=flower)
                mask_uint8 = (mask * 255).astype(np.uint8)
                flower_mask.mask = self.bridge.cv2_to_imgmsg(mask_uint8, encoding="mono8")
                
                flower_masks_msg.flowers.append(flower_mask)
            
            self.get_logger().debug(f"Detected {len(flower_masks_msg.flowers)} flowers")
            self.pub.publish(flower_masks_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
