#!/usr/bin/env python3
"""Local refinement node (Phase 2 — pruning).

Subscribes to end-effector color and depth cameras.
Re-runs flower detection at close range.
Merges with coarse map landmarks.
Publishes refined pruning targets.
"""

import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from geometry_msgs.msg import PointStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point

from oskar_msgs.msg import FlowerLandmark, PruningTargets, ThinningPlan


class LocalRefinementNode(Node):
    def __init__(self):
        super().__init__("local_refinement_node")

        # Declare parameters
        self.declare_parameter("use_sim_time", False)
        self.declare_parameter(
            "model_path",
            "/home/workstation/oskar/2025-transformers-for-apple-flower-segmentation-bhangale/"
            "experiments/training_outputs/maskrcnn/test_batch/best_hp_run/finetune/model_final.pth",
        )
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("merge_distance_m", 0.05)

        # Get parameters
        model_path = self.get_parameter("model_path").value
        self.confidence_threshold = self.get_parameter(
            "confidence_threshold"
        ).value
        device = self.get_parameter("device").value
        self.merge_distance = self.get_parameter("merge_distance_m").value

        # Load model
        self.predictor = self.load_model(model_path, device)

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ROS communication
        qos_profile = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.BEST_EFFORT
        )

        # Subscribe to end-effector camera
        ee_color_sub = Subscriber(
            self, Image, "/ee_camera/image_color", qos_profile=qos_profile
        )
        ee_depth_sub = Subscriber(
            self, Image, "/ee_camera/depth", qos_profile=qos_profile
        )
        ee_info_sub = Subscriber(
            self, CameraInfo, "/ee_camera/camera_info", qos_profile=qos_profile
        )

        # Synchronize
        self.ts = ApproximateTimeSynchronizer(
            [ee_color_sub, ee_depth_sub, ee_info_sub],
            queue_size=10,
            slop=0.05,
        )
        self.ts.registerCallback(self.camera_callback)

        # Subscribe to thinning plan
        self.plan = None
        self.create_subscription(
            ThinningPlan,
            "/orchard/thinning_plan",
            self.plan_callback,
            qos_profile,
        )

        # Publisher
        self.pub = self.create_publisher(
            PruningTargets, "/pruning/targets", qos_profile
        )

        self.bridge = CvBridge()
        self.get_logger().info("LocalRefinementNode initialized")

    def load_model(self, model_path, device):
        """Load Detectron2 model."""
        from pathlib import Path
        try:
            cfg = get_cfg()
            config_path = Path(model_path).parent / "detectron2_config.yaml"
            cfg.merge_from_file(str(config_path))
            cfg.MODEL.WEIGHTS = str(model_path)
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
            cfg.MODEL.DEVICE = device
            return DefaultPredictor(cfg)
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise

    def plan_callback(self, msg):
        """Store thinning plan."""
        self.plan = msg

    def camera_callback(self, color_msg, depth_msg, info_msg):
        """Process end-effector camera."""
        try:
            if self.plan is None:
                return

            # Convert color to OpenCV
            color_cv = self.bridge.imgmsg_to_cv2(
                color_msg, desired_encoding="bgr8"
            )
            depth_cv = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="32FC1"
            )

            # Run inference
            outputs = self.predictor(color_cv)
            instances = outputs["instances"]
            masks = instances.pred_masks.cpu().numpy()
            scores = instances.scores.cpu().numpy()

            # Extract intrinsics
            K = np.array(info_msg.k).reshape(3, 3)
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]

            # Process detections
            ee_landmarks = []
            for instance_id, (mask, score) in enumerate(zip(masks, scores)):
                if score < self.confidence_threshold:
                    continue

                # Compute centroid
                y_coords, x_coords = np.where(mask)
                if len(x_coords) == 0:
                    continue
                centroid_u = float(np.mean(x_coords))
                centroid_v = float(np.mean(y_coords))

                # Sample median depth
                depth_samples = depth_cv[mask > 0]
                valid_depths = depth_samples[~np.isnan(depth_samples)]
                if len(valid_depths) < 5:
                    continue
                median_depth = float(np.median(valid_depths))

                # Backproject to 3D
                x_cam = (centroid_u - cx) / fx * median_depth
                y_cam = (centroid_v - cy) / fy * median_depth
                z_cam = median_depth

                # Transform to map frame
                try:
                    point_cam = PointStamped()
                    point_cam.header.stamp = color_msg.header.stamp
                    point_cam.header.frame_id = (
                        color_msg.header.frame_id
                    )  # e.g., 'ee_camera'
                    point_cam.point.x = x_cam
                    point_cam.point.y = y_cam
                    point_cam.point.z = z_cam

                    transform = self.tf_buffer.lookup_transform(
                        "map",
                        color_msg.header.frame_id,
                        color_msg.header.stamp,
                        timeout=rclpy.duration.Duration(seconds=0.1),
                    )
                    point_map = do_transform_point(point_cam, transform)

                    landmark = FlowerLandmark()
                    landmark.landmark_id = instance_id
                    landmark.position = point_map.point
                    landmark.observation_count = 1
                    landmark.mean_confidence = float(score)
                    landmark.is_inferred = False
                    landmark.tree_id = 0

                    ee_landmarks.append(landmark)
                except Exception as e:
                    self.get_logger().warn(f"Transform failed: {e}")
                    continue

            # Create pruning targets (placeholder for now)
            # In a full implementation, merge with coarse map and rank
            if self.plan and len(self.plan.tree_ids) > 0:
                targets = PruningTargets()
                targets.header.stamp = color_msg.header.stamp
                targets.header.frame_id = "map"
                targets.tree_id = self.plan.tree_ids[0]  # First tree
                targets.current_count = len(ee_landmarks)
                targets.target_count = max(
                    1, targets.current_count - self.plan.rough_removal_count[0]
                )
                targets.ranked_targets = ee_landmarks

                self.pub.publish(targets)

        except Exception as e:
            self.get_logger().error(f"Error in camera_callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = LocalRefinementNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
