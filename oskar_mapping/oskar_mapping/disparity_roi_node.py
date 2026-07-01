#!/usr/bin/env python3
"""Stereo disparity estimation node using Fast-FoundationStereo on flower ROIs.

Subscribes to StereoPair and FlowerMasks.
Synchronizes them using ApproximateTimeSynchronizer.
Runs FFM model only on the crop regions of detected flowers.
Publishes DisparityImage, depth map, and point cloud (32FC1).
"""

# Scrub statistics module shadowing (workaround for ROS2 statistics package shadow)
import sys as _sys
_sys.path = [p for p in _sys.path if "/install/statistics/" not in p]

from pathlib import Path
import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2, PointField
from stereo_msgs.msg import DisparityImage

from oskar_msgs.msg import StereoPair, FlowerMasks


class FastFoundationStereoModel:
    """Wrapper around Fast-FoundationStereo model."""

    def __init__(self, ffm_dir, model_path, valid_iters=8, max_disp=192):
        import sys

        ffm_dir = str(ffm_dir)
        if ffm_dir not in sys.path:
            sys.path.insert(0, ffm_dir)

        # InputPadder must be imported after sys.path is set
        from core.utils.utils import InputPadder

        self.InputPadder = InputPadder

        self.valid_iters = valid_iters
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )

        torch.autograd.set_grad_enabled(False)
        self.model = torch.load(
            str(model_path),
            map_location="cpu",
            weights_only=False,
        )
        self.model.args.valid_iters = valid_iters
        self.model.args.max_disp = max_disp
        self.model = self.model.to(self.device).eval()

    def forward(self, left_bgr, right_bgr):
        """left_bgr, right_bgr: numpy (H, W, 3), BGR uint8 from OpenCV.
        Returns: disparity map (H, W), float32.
        """
        H, W = left_bgr.shape[:2]
        # FFM was trained on RGB images
        left_rgb = left_bgr[..., ::-1].copy()
        right_rgb = right_bgr[..., ::-1].copy()

        img0 = (
            torch.as_tensor(left_rgb)
            .to(self.device)
            .float()[None]
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        img1 = (
            torch.as_tensor(right_rgb)
            .to(self.device)
            .float()[None]
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        padder = self.InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.amp.autocast(
            "cuda",
            enabled=(self.device.type == "cuda"),
            dtype=torch.float16,
        ):
            disp = self.model.forward(
                img0,
                img1,
                iters=self.valid_iters,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )

        disp = padder.unpad(disp.float())
        return (
            disp.data.cpu()
            .numpy()
            .reshape(H, W)
            .clip(0, None)
            .astype(np.float32)
        )

    def forward_batch(self, left_bgr_list, right_bgr_list):
        """left_bgr_list, right_bgr_list: lists of numpy (H, W, 3), BGR uint8 from OpenCV.
        Returns: list of disparity maps (H, W), float32.
        """
        if not left_bgr_list:
            return []

        B = len(left_bgr_list)
        H, W = left_bgr_list[0].shape[:2]

        left_rgb_list = [img[..., ::-1].copy() for img in left_bgr_list]
        right_rgb_list = [img[..., ::-1].copy() for img in right_bgr_list]

        img0 = (
            torch.as_tensor(np.stack(left_rgb_list))
            .to(self.device)
            .float()
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        img1 = (
            torch.as_tensor(np.stack(right_rgb_list))
            .to(self.device)
            .float()
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        padder = self.InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.amp.autocast(
            "cuda",
            enabled=(self.device.type == "cuda"),
            dtype=torch.float16,
        ):
            disp = self.model.forward(
                img0,
                img1,
                iters=self.valid_iters,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )

        disp = padder.unpad(disp.float())
        disp_np = (
            disp.data.cpu()
            .numpy()
            .clip(0, None)
            .astype(np.float32)
        )
        return [disp_np[i, 0] for i in range(B)]


class DisparityROINode(Node):
    def __init__(self):
        super().__init__("disparity_roi_node")

        # Declare parameters
        self.declare_parameter(
            "ffm_dir",
            "/home/workstation/oskar/Fast-FoundationStereo",
        )
        self.declare_parameter(
            "model_dir",
            "/home/workstation/oskar/Fast-FoundationStereo/weights/"
            "23-36-37/model_best_bp2_serialize.pth",
        )
        self.declare_parameter("valid_iters", 8)
        self.declare_parameter("max_disp", 384)
        self.declare_parameter("padding_x", 16)
        self.declare_parameter("padding_y", 16)
        self.declare_parameter("roi_height", 256)
        self.declare_parameter("roi_width", 512)
        self.declare_parameter("batch_size", 16)
        self.declare_parameter("baseline_m", 0.0)

        # Get parameters
        ffm_dir = self.get_parameter("ffm_dir").value
        model_dir = self.get_parameter("model_dir").value
        valid_iters = self.get_parameter("valid_iters").value
        self.max_disp = self.get_parameter("max_disp").value
        self.padding_x = self.get_parameter("padding_x").value
        self.padding_y = self.get_parameter("padding_y").value
        self.roi_height = self.get_parameter("roi_height").value
        self.roi_width = self.get_parameter("roi_width").value
        self.batch_size = self.get_parameter("batch_size").value
        self._baseline_override = self.get_parameter("baseline_m").value

        self.get_logger().info(f"FFM dir: {ffm_dir}")
        self.get_logger().info(f"Model: {model_dir}")
        self.get_logger().info(f"Valid iters: {valid_iters}, Max disparity: {self.max_disp}")
        self.get_logger().info(f"Padding x: {self.padding_x}, Padding y: {self.padding_y}")
        self.get_logger().info(f"ROI size: height={self.roi_height}, width={self.roi_width}")
        self.get_logger().info(f"Batch size: {self.batch_size}")

        if not Path(model_dir).exists():
            self.get_logger().error(
                f"FFM checkpoint not found: {model_dir}\n"
                f"Download it to {ffm_dir}/weights/23-36-37/ from the Fast-FoundationStereo release.",
            )
            raise FileNotFoundError(model_dir)

        self.model = FastFoundationStereoModel(
            ffm_dir,
            model_dir,
            valid_iters,
            self.max_disp,
        )

        # ROS communication
        # Subscriptions: use RELIABLE QoS to avoid dropping large messages
        sub_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        
        self.sync_pair_sub = Subscriber(self, StereoPair, "/stereo/sync_pair", qos_profile=sub_qos)
        self.masks_sub = Subscriber(self, FlowerMasks, "/flowers/masks", qos_profile=sub_qos)

        self.ts = ApproximateTimeSynchronizer(
            [self.sync_pair_sub, self.masks_sub],
            queue_size=10,
            slop=0.05
        )
        self.ts.registerCallback(self.sync_callback)

        # Publishers: standard topics
        self.pub_disparity = self.create_publisher(DisparityImage, "/stereo/disparity", 10)
        self.pub_depth = self.create_publisher(Image, "/stereo/depth", 10)
        self.pub_cloud = self.create_publisher(PointCloud2, "/stereo/points", 10)

        self.bridge = CvBridge()
        self.get_logger().info("DisparityROINode initialized")

    def sync_callback(self, stereo_pair_msg, masks_msg):
        """Process synchronized StereoPair and FlowerMasks, computing disparity only on flower ROIs."""
        try:
            left_cv = self.bridge.imgmsg_to_cv2(
                stereo_pair_msg.left_image,
                desired_encoding="bgr8",
            )
            right_cv = self.bridge.imgmsg_to_cv2(
                stereo_pair_msg.right_image,
                desired_encoding="bgr8",
            )

            height, width = left_cv.shape[:2]
            disparity = np.zeros((height, width), dtype=np.float32)

            num_flowers = len(masks_msg.flowers)
            self.get_logger().info(f"Received sync pair: processing {num_flowers} flowers.")

            left_crops = []
            right_crops = []
            crop_meta = []

            for flower in masks_msg.flowers:
                mask_cv = self.bridge.imgmsg_to_cv2(flower.mask, desired_encoding="mono8")
                mask_bool = mask_cv > 127

                y_indices, x_indices = np.where(mask_bool)
                if len(x_indices) == 0:
                    continue

                ymin, ymax = y_indices.min(), y_indices.max()
                xmin, xmax = x_indices.min(), x_indices.max()

                # Target crop dimensions: fixed to self.roi_height and self.roi_width
                # This ensures consistent input shape to PyTorch, avoiding recompilations
                h_target = self.roi_height
                w_target = self.roi_width

                if h_target > height or w_target > width:
                    self.get_logger().warn(
                        f"ROI target size ({h_target}x{w_target}) exceeds image bounds ({height}x{width}). Slicing defaults.",
                        throttle_duration_sec=5.0
                    )
                    continue

                # Vertical centering
                y_center = (ymin + ymax) // 2
                ymin_padded = y_center - h_target // 2
                ymax_padded = ymin_padded + h_target

                if ymin_padded < 0:
                    ymin_padded = 0
                    ymax_padded = h_target
                elif ymax_padded > height:
                    ymax_padded = height
                    ymin_padded = height - h_target

                # Horizontal alignment: flower is on the right, search space is on the left
                # Target horizontal span to cover is [xmin - max_disp, xmax]
                L = xmin - self.max_disp - self.padding_x
                R = xmax + self.padding_x
                
                if (R - L) <= w_target:
                    # Center the span in the target width
                    x_center = (L + R) // 2
                    xmin_expanded = x_center - w_target // 2
                    xmax_expanded = xmin_expanded + w_target
                else:
                    # Priority is to align with the right side of the flower
                    xmax_expanded = R
                    xmin_expanded = xmax_expanded - w_target

                if xmin_expanded < 0:
                    xmin_expanded = 0
                    xmax_expanded = w_target
                elif xmax_expanded > width:
                    xmax_expanded = width
                    xmin_expanded = width - w_target

                # Extract contiguous crops
                left_tile = np.ascontiguousarray(left_cv[ymin_padded:ymax_padded, xmin_expanded:xmax_expanded])
                right_tile = np.ascontiguousarray(right_cv[ymin_padded:ymax_padded, xmin_expanded:xmax_expanded])

                left_crops.append(left_tile)
                right_crops.append(right_tile)
                crop_meta.append((ymin_padded, ymax_padded, xmin_expanded, xmax_expanded, mask_bool))

            # Run inference in batches to optimize GPU speed and prevent OOM
            batch_size = self.batch_size
            for i in range(0, len(left_crops), batch_size):
                left_batch = left_crops[i : i + batch_size]
                right_batch = right_crops[i : i + batch_size]
                meta_batch = crop_meta[i : i + batch_size]

                # Model batch forward
                tile_disps = self.model.forward_batch(left_batch, right_batch)

                # Paste back
                for tile_disp, (ymin_p, ymax_p, xmin_e, xmax_e, mask_b) in zip(tile_disps, meta_batch):
                    crop_mask = mask_b[ymin_p:ymax_p, xmin_e:xmax_e]
                    if tile_disp.shape == crop_mask.shape:
                        disparity[ymin_p:ymax_p, xmin_e:xmax_e] = np.where(
                            crop_mask,
                            tile_disp,
                            disparity[ymin_p:ymax_p, xmin_e:xmax_e]
                        )

            # Depth from disparity: d = (fx * baseline) / disparity
            fx = stereo_pair_msg.left_info.k[0]
            p3 = stereo_pair_msg.right_info.p[3]
            p0 = stereo_pair_msg.right_info.p[0]
            baseline = (
                (-p3 / p0)
                if (p0 != 0 and p3 != 0)
                else self._baseline_override
            )
            if baseline <= 0:
                self.get_logger().warn(
                    "Baseline is zero — set disparity_roi_node.baseline_m in the config.",
                    throttle_duration_sec=5.0,
                )
                return

            depth = np.full_like(disparity, np.nan)
            valid = disparity > 0
            depth[valid] = (fx * baseline) / disparity[valid]

            # Publish DisparityImage
            disparity_msg = DisparityImage()
            disparity_msg.header.stamp = stereo_pair_msg.header.stamp
            disparity_msg.header.frame_id = stereo_pair_msg.header.frame_id
            disparity_msg.image = self.bridge.cv2_to_imgmsg(
                disparity,
                encoding="32FC1",
            )
            disparity_msg.f = float(fx)
            disparity_msg.t = float(baseline)
            disparity_msg.valid_window.x_offset = 0
            disparity_msg.valid_window.y_offset = 0
            disparity_msg.valid_window.height = disparity.shape[0]
            disparity_msg.valid_window.width = disparity.shape[1]
            valid_disp = disparity[valid]
            disparity_msg.min_disparity = (
                float(valid_disp.min()) if valid.any() else 0.0
            )
            disparity_msg.max_disparity = (
                float(valid_disp.max()) if valid.any() else 0.0
            )
            disparity_msg.delta_d = 1.0
            self.pub_disparity.publish(disparity_msg)

            # Publish depth
            depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
            depth_msg.header.stamp = stereo_pair_msg.header.stamp
            depth_msg.header.frame_id = stereo_pair_msg.header.frame_id
            self.pub_depth.publish(depth_msg)

            # Publish point cloud
            cloud_msg = self._depth_to_pointcloud(
                depth,
                stereo_pair_msg.left_info,
                stereo_pair_msg.header.stamp,
                stereo_pair_msg.header.frame_id,
            )
            self.pub_cloud.publish(cloud_msg)

        except Exception as e:
            self.get_logger().error(f"Error in sync_callback: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _depth_to_pointcloud(self, depth, info, stamp, frame_id):
        fx = float(info.k[0])
        fy = float(info.k[4])
        cx = float(info.k[2])
        cy = float(info.k[5])
        H, W = depth.shape

        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)

        valid = np.isfinite(depth) & (depth > 0.0)
        z = depth[valid]
        x = (uu[valid] - cx) * z / fx
        y = (vv[valid] - cy) * z / fy

        pts = np.stack([x, y, z], axis=1).astype(np.float32)

        cloud = PointCloud2()
        cloud.header.stamp = stamp
        cloud.header.frame_id = frame_id
        cloud.height = 1
        cloud.width = len(pts)
        cloud.is_dense = True
        cloud.is_bigendian = False
        cloud.fields = [
            PointField(
                name="x",
                offset=0,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="y",
                offset=4,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="z",
                offset=8,
                datatype=PointField.FLOAT32,
                count=1,
            ),
        ]
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * cloud.width
        cloud.data = pts.tobytes()
        return cloud


def main(args=None):
    rclpy.init(args=args)
    node = DisparityROINode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
