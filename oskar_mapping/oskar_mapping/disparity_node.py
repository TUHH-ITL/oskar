#!/usr/bin/env python3
"""Stereo disparity estimation node using Fast-FoundationStereo.

Subscribes to rectified StereoPair.
Publishes DisparityImage and depth map (32FC1).
"""

# The ROS workspace installs a 'statistics' package under install/statistics/
# that shadows the Python stdlib statistics module.  torch._inductor imports
# statistics.median at startup, so we must scrub the shadow before torch loads.
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
from sensor_msgs.msg import Image, PointCloud2, PointField
from stereo_msgs.msg import DisparityImage

from oskar_msgs.msg import StereoPair


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


class DisparityNode(Node):
    def __init__(self):
        super().__init__("disparity_node")

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
        self.declare_parameter("scale", 1.0)
        self.declare_parameter("valid_iters", 8)
        self.declare_parameter("max_disp", 192)
        self.declare_parameter("tile_height", 0)
        self.declare_parameter("tile_overlap", 64)
        self.declare_parameter("tile_width", 0)
        self.declare_parameter("tile_x_overlap", 0)
        # Explicit baseline override — required when Isaac Sim does not encode
        # the stereo baseline in CameraInfo.P[0][3] (which it never does for
        # single-camera omnigraph nodes).  Set to the physical camera separation
        # in metres.  Ignored if CameraInfo.P[0][3] already encodes a valid
        # baseline.
        self.declare_parameter("baseline_m", 0.0)

        # Get parameters
        ffm_dir = self.get_parameter("ffm_dir").value
        model_dir = self.get_parameter("model_dir").value
        self.scale = self.get_parameter("scale").value
        valid_iters = self.get_parameter("valid_iters").value
        max_disp = self.get_parameter("max_disp").value
        self.tile_height = int(self.get_parameter("tile_height").value)
        self.tile_overlap = int(self.get_parameter("tile_overlap").value)
        self.tile_width = int(self.get_parameter("tile_width").value)
        self.tile_x_overlap = int(self.get_parameter("tile_x_overlap").value)
        self._baseline_override = self.get_parameter("baseline_m").value
        if max_disp > 416:
            self.get_logger().warn(
                "max_disp > 416 can exceed this FFM checkpoint's positional embedding. "
                "Use max_disp:=384 unless you have a checkpoint trained for a larger range.",
            )
        if self.tile_width > 0 and self.tile_x_overlap <= 0:
            self.tile_x_overlap = max_disp

        self.get_logger().info(f"FFM dir: {ffm_dir}")
        self.get_logger().info(f"Model: {model_dir}")
        self.get_logger().info(
            f"Scale: {self.scale}, valid_iters: {valid_iters}",
        )
        if self.tile_height > 0 or self.tile_width > 0:
            self.get_logger().info(
                f"Tiled inference enabled: tile_height={self.tile_height}, "
                f"tile_overlap={self.tile_overlap}, tile_width={self.tile_width}, "
                f"tile_x_overlap={self.tile_x_overlap}",
            )

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
            max_disp,
        )
        self.warmup_model()

        # ROS communication
        # Subscriber: RELIABLE — the StereoPair is large (~9 MB); BEST_EFFORT silently
        # drops the fragmented message on localhost, starving this node. RELIABLE
        # requests retransmission so every frame arrives (matches the RELIABLE publisher).
        sub_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.sub = self.create_subscription(
            StereoPair,
            "/stereo/sync_pair",
            self.stereo_callback,
            sub_qos,
        )
        # Publishers: RELIABLE so RViz and downstream nodes can subscribe without QoS warnings
        self.pub_disparity = self.create_publisher(
            DisparityImage,
            "/stereo/disparity",
            10,
        )
        self.pub_depth = self.create_publisher(Image, "/stereo/depth", 10)
        self.pub_cloud = self.create_publisher(
            PointCloud2,
            "/stereo/points",
            10,
        )

        self.bridge = CvBridge()
        self.get_logger().info("DisparityNode initialized")

    def warmup_model(self):
        """Warm up model at the configured (post-scale) resolution to trigger CUDA compilation."""
        self.get_logger().info(
            "Warming up disparity model (first run triggers CUDA compilation, ~30s)...",
        )
        try:
            h = int(720 * self.scale)
            w = int(1280 * self.scale)
            dummy_left = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            dummy_right = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            _ = self.model.forward(dummy_left, dummy_right)
            self.get_logger().info("Model warm-up complete")
        except Exception as e:
            self.get_logger().warn(f"Model warm-up failed: {e}")

    def stereo_callback(self, stereo_pair_msg):
        """Process stereo pair and compute disparity."""
        try:
            left_cv = self.bridge.imgmsg_to_cv2(
                stereo_pair_msg.left_image,
                desired_encoding="bgr8",
            )
            right_cv = self.bridge.imgmsg_to_cv2(
                stereo_pair_msg.right_image,
                desired_encoding="bgr8",
            )

            # Scale if needed (real robot: 20 MP → ~2.7 MP for inference)
            if self.scale < 1.0:
                h_scaled = int(left_cv.shape[0] * self.scale)
                w_scaled = int(left_cv.shape[1] * self.scale)
                left_scaled = cv2.resize(
                    left_cv,
                    (w_scaled, h_scaled),
                    interpolation=cv2.INTER_AREA,
                )
                right_scaled = cv2.resize(
                    right_cv,
                    (w_scaled, h_scaled),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                left_scaled = left_cv
                right_scaled = right_cv

            disparity = self._compute_disparity(left_scaled, right_scaled)

            # Upsample back to original resolution (INTER_NEAREST preserves disparity values)
            if self.scale < 1.0:
                disparity = cv2.resize(
                    disparity,
                    (left_cv.shape[1], left_cv.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
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
                    "Baseline is zero — set disparity_node.baseline_m in the config.",
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
            self.get_logger().error(f"Error in stereo_callback: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _compute_disparity(self, left, right):
        need_tiling_y = (self.tile_height > 0 and self.tile_height < left.shape[0])
        need_tiling_x = (self.tile_width > 0 and self.tile_width < left.shape[1])
        if not need_tiling_y and not need_tiling_x:
            return self.model.forward(left, right)
        return self._compute_disparity_tiled(left, right)

    def _compute_disparity_tiled(self, left, right):
        height, width = left.shape[:2]

        tile_height = self.tile_height if (self.tile_height > 0 and self.tile_height < height) else height
        tile_width = self.tile_width if (self.tile_width > 0 and self.tile_width < width) else width

        overlap_y = max(0, min(self.tile_overlap, tile_height // 2 - 1))
        step_y = tile_height - 2 * overlap_y
        if step_y <= 0:
            step_y = tile_height
            overlap_y = 0

        # tile_x_overlap must be at least max_disp to avoid losing the search range near boundaries
        max_disp = self.model.model.args.max_disp
        overlap_x = self.tile_x_overlap if self.tile_x_overlap > 0 else max_disp
        if overlap_x >= tile_width:
            overlap_x = tile_width // 2
        step_x = tile_width - overlap_x
        if step_x <= 0:
            step_x = tile_width
            overlap_x = 0

        disparity = np.zeros((height, width), dtype=np.float32)
        filled = np.zeros((height, width), dtype=bool)

        y0 = 0
        while y0 < height:
            core_start_y = y0
            core_end_y = min(y0 + step_y, height)

            tile_start_y = max(0, core_start_y - overlap_y)
            tile_end_y = min(height, tile_start_y + tile_height)
            if tile_end_y > height:
                tile_end_y = height
                tile_start_y = max(0, tile_end_y - tile_height)

            x0 = 0
            while x0 < width:
                core_start_x = x0
                core_end_x = min(x0 + step_x, width)

                tile_start_x = max(0, core_start_x - overlap_x)
                tile_end_x = min(width, tile_start_x + tile_width)
                if tile_end_x > width:
                    tile_end_x = width
                    tile_start_x = max(0, tile_end_x - tile_width)

                left_tile = np.ascontiguousarray(left[tile_start_y:tile_end_y, tile_start_x:tile_end_x])
                right_tile = np.ascontiguousarray(right[tile_start_y:tile_end_y, tile_start_x:tile_end_x])
                tile_disp = self.model.forward(left_tile, right_tile)

                src_start_y = core_start_y - tile_start_y
                src_end_y = src_start_y + (core_end_y - core_start_y)

                src_start_x = core_start_x - tile_start_x
                src_end_x = src_start_x + (core_end_x - core_start_x)

                disparity[core_start_y:core_end_y, core_start_x:core_end_x] = tile_disp[src_start_y:src_end_y, src_start_x:src_end_x]
                filled[core_start_y:core_end_y, core_start_x:core_end_x] = True

                self.get_logger().info(
                    f"Processed tile y:{tile_start_y}-{tile_end_y}, x:{tile_start_x}-{tile_end_x} "
                    f"(stitched core y:{core_start_y}-{core_end_y}, x:{core_start_x}-{core_end_x})",
                    throttle_duration_sec=5.0,
                )
                x0 = core_end_x
            y0 = core_end_y

        if not filled.all():
            self.get_logger().warn("Some pixels in the disparity image were not filled during tiling.")
        return disparity

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
    node = DisparityNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
