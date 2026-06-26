#!/usr/bin/env python3
"""
Stereo disparity estimation node using Fast-FoundationStereo.

Subscribes to rectified StereoPair.
Publishes DisparityImage and depth map (32FC1).
"""

# The ROS workspace installs a 'statistics' package under install/statistics/
# that shadows the Python stdlib statistics module.  torch._inductor imports
# statistics.median at startup, so we must scrub the shadow before torch loads.
import sys as _sys
_sys.path = [p for p in _sys.path if '/install/statistics/' not in p]

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import cv2
import numpy as np
from cv_bridge import CvBridge
import torch
from pathlib import Path

from oskar_msgs.msg import StereoPair
from stereo_msgs.msg import DisparityImage
from sensor_msgs.msg import Image, PointCloud2, PointField


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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        torch.autograd.set_grad_enabled(False)
        self.model = torch.load(str(model_path), map_location='cpu', weights_only=False)
        self.model.args.valid_iters = valid_iters
        self.model.args.max_disp = max_disp
        self.model = self.model.to(self.device).eval()

    def forward(self, left_bgr, right_bgr):
        """
        left_bgr, right_bgr: numpy (H, W, 3), BGR uint8 from OpenCV.
        Returns: disparity map (H, W), float32.
        """
        H, W = left_bgr.shape[:2]
        # FFM was trained on RGB images
        left_rgb = left_bgr[..., ::-1].copy()
        right_rgb = right_bgr[..., ::-1].copy()

        img0 = torch.as_tensor(left_rgb).to(self.device).float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_rgb).to(self.device).float()[None].permute(0, 3, 1, 2)

        padder = self.InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda'), dtype=torch.float16):
            disp = self.model.forward(img0, img1, iters=self.valid_iters, test_mode=True,
                                      optimize_build_volume='pytorch1')

        disp = padder.unpad(disp.float())
        return disp.data.cpu().numpy().reshape(H, W).clip(0, None).astype(np.float32)


class DisparityNode(Node):
    def __init__(self):
        super().__init__('disparity_node')

        # Declare parameters
        self.declare_parameter('ffm_dir', '/home/workstation/oskar/Fast-FoundationStereo')
        self.declare_parameter('model_dir',
                               '/home/workstation/oskar/Fast-FoundationStereo/weights/'
                               '23-36-37/model_best_bp2_serialize.pth')
        self.declare_parameter('scale', 1.0)
        self.declare_parameter('valid_iters', 8)
        self.declare_parameter('max_disp', 192)
        # Explicit baseline override — required when Isaac Sim does not encode
        # the stereo baseline in CameraInfo.P[0][3] (which it never does for
        # single-camera omnigraph nodes).  Set to the physical camera separation
        # in metres.  Ignored if CameraInfo.P[0][3] already encodes a valid
        # baseline.
        self.declare_parameter('baseline_m', 0.0)

        # Get parameters
        ffm_dir = self.get_parameter('ffm_dir').value
        model_dir = self.get_parameter('model_dir').value
        self.scale = self.get_parameter('scale').value
        valid_iters = self.get_parameter('valid_iters').value
        max_disp = self.get_parameter('max_disp').value
        self._baseline_override = self.get_parameter('baseline_m').value

        self.get_logger().info(f"FFM dir: {ffm_dir}")
        self.get_logger().info(f"Model: {model_dir}")
        self.get_logger().info(f"Scale: {self.scale}, valid_iters: {valid_iters}")

        if not Path(model_dir).exists():
            self.get_logger().error(
                f"FFM checkpoint not found: {model_dir}\n"
                f"Download it to {ffm_dir}/weights/23-36-37/ from the Fast-FoundationStereo release."
            )
            raise FileNotFoundError(model_dir)

        self.model = FastFoundationStereoModel(ffm_dir, model_dir, valid_iters, max_disp)
        self.warmup_model()

        # ROS communication
        # Subscriber: RELIABLE — the StereoPair is large (~9 MB); BEST_EFFORT silently
        # drops the fragmented message on localhost, starving this node. RELIABLE
        # requests retransmission so every frame arrives (matches the RELIABLE publisher).
        sub_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.sub = self.create_subscription(StereoPair, '/stereo/sync_pair',
                                            self.stereo_callback, sub_qos)
        # Publishers: RELIABLE so RViz and downstream nodes can subscribe without QoS warnings
        self.pub_disparity = self.create_publisher(DisparityImage, '/stereo/disparity', 10)
        self.pub_depth     = self.create_publisher(Image,          '/stereo/depth',      10)
        self.pub_cloud     = self.create_publisher(PointCloud2,    '/stereo/points',     10)

        self.bridge = CvBridge()
        self.get_logger().info("DisparityNode initialized")

    def warmup_model(self):
        """Warm up model at the configured (post-scale) resolution to trigger CUDA compilation."""
        self.get_logger().info(
            "Warming up disparity model (first run triggers CUDA compilation, ~30s)...")
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
            left_cv = self.bridge.imgmsg_to_cv2(stereo_pair_msg.left_image,
                                                 desired_encoding="bgr8")
            right_cv = self.bridge.imgmsg_to_cv2(stereo_pair_msg.right_image,
                                                  desired_encoding="bgr8")

            # Scale if needed (real robot: 20 MP → ~2.7 MP for inference)
            if self.scale < 1.0:
                h_scaled = int(left_cv.shape[0] * self.scale)
                w_scaled = int(left_cv.shape[1] * self.scale)
                left_scaled = cv2.resize(left_cv, (w_scaled, h_scaled),
                                         interpolation=cv2.INTER_AREA)
                right_scaled = cv2.resize(right_cv, (w_scaled, h_scaled),
                                          interpolation=cv2.INTER_AREA)
            else:
                left_scaled = left_cv
                right_scaled = right_cv

            disparity = self.model.forward(left_scaled, right_scaled)

            # Upsample back to original resolution (INTER_NEAREST preserves disparity values)
            if self.scale < 1.0:
                disparity = cv2.resize(disparity,
                                       (left_cv.shape[1], left_cv.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)

            # Depth from disparity: d = (fx * baseline) / disparity
            fx = stereo_pair_msg.left_info.k[0]
            p3 = stereo_pair_msg.right_info.p[3]
            p0 = stereo_pair_msg.right_info.p[0]
            baseline = (-p3 / p0) if (p0 != 0 and p3 != 0) else self._baseline_override
            if baseline <= 0:
                self.get_logger().warn(
                    'Baseline is zero — set disparity_node.baseline_m in the config.',
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
            disparity_msg.image = self.bridge.cv2_to_imgmsg(disparity, encoding="32FC1")
            disparity_msg.f = float(fx)
            disparity_msg.t = float(baseline)
            disparity_msg.valid_window.x_offset = 0
            disparity_msg.valid_window.y_offset = 0
            disparity_msg.valid_window.height = disparity.shape[0]
            disparity_msg.valid_window.width = disparity.shape[1]
            valid_disp = disparity[valid]
            disparity_msg.min_disparity = float(valid_disp.min()) if valid.any() else 0.0
            disparity_msg.max_disparity = float(valid_disp.max()) if valid.any() else 0.0
            disparity_msg.delta_d = 1.0
            self.pub_disparity.publish(disparity_msg)

            # Publish depth
            depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
            depth_msg.header.stamp = stereo_pair_msg.header.stamp
            depth_msg.header.frame_id = stereo_pair_msg.header.frame_id
            self.pub_depth.publish(depth_msg)

            # Publish point cloud
            cloud_msg = self._depth_to_pointcloud(
                depth, stereo_pair_msg.left_info,
                stereo_pair_msg.header.stamp,
                stereo_pair_msg.header.frame_id,
            )
            self.pub_cloud.publish(cloud_msg)

        except Exception as e:
            self.get_logger().error(f"Error in stereo_callback: {e}")

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
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
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


if __name__ == '__main__':
    main()
