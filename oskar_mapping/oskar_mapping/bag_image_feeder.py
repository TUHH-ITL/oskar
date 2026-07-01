#!/usr/bin/env python3
"""Image feeder for the Fraunhofer IFAM apple-orchard dataset.

The dataset's stereo images are NOT in the rosbag — they are JPG files inside two
zip folders (top = SAMSON3, bottom = SAMSON4), already extracted to a directory.
Each file is named ``{frameid}_{epoch_seconds}-{milliseconds}.jpg`` and the two
cameras share the same ``frameid`` for a stereo pair.

This node reads the matched pairs and republishes them as ROS Image messages on the
topics the rest of the Phase 1 pipeline expects, plus a CameraInfo so
``stereo_sync_node`` has calibration to work with.

CALIBRATION: pass real per-camera stereo calibration via ``left_calib_file`` /
``right_calib_file`` (the SAMSON3/SAMSON4 ``*_stereo.yaml`` files: cameraMatrix,
distCoeffs, rotation, projectionMatrix at full sensor resolution). The intrinsics are
scaled automatically to the downscaled image size; distortion + rectification rotation
are passed through unchanged so ``stereo_sync_node`` performs a real undistort+rectify
and depth becomes metric. If no calib file is given, the node falls back to a
*synthesized* CameraInfo (guessed fx, zero distortion, identity rectification) — fine
for testing the image path / segmentation, but depth is NOT metric.
"""

import glob
import os

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


def _repo_root():
    """Return the repository root containing `oskar_mapping/` and `docs/`."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _default_calib_path(name):
    """Return a bundled stereo calibration path if it exists."""
    path = os.path.join(_repo_root(), "oskar_mapping", "bagfile_data", name)
    return path if os.path.exists(path) else ""


def _frame_id_and_stamp(path):
    """Parse '003396_1713171581-898.jpg' -> (frameid='003396', sec, nanosec)."""
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    frameid, ts = name.split("_", 1)
    sec_str, ms_str = ts.split("-")
    return frameid, int(sec_str), int(ms_str) * 1_000_000


class BagImageFeederNode(Node):
    def __init__(self):
        super().__init__("bag_image_feeder_node")

        self.declare_parameter("top_dir", "")
        self.declare_parameter("bottom_dir", "")
        self.declare_parameter("left_image_topic", "/rgb_static_1")
        self.declare_parameter("right_image_topic", "/rgb_static_2")
        self.declare_parameter("left_info_topic", "/static_camera_info_1")
        self.declare_parameter("right_info_topic", "/static_camera_info_2")
        self.declare_parameter("left_camera_frame", "static_camera_1")
        self.declare_parameter("right_camera_frame", "static_camera_2")
        self.declare_parameter("rate_hz", 2.0)
        self.declare_parameter("scale", 1.0)          # downscale factor for testing
        self.declare_parameter("loop", False)
        self.declare_parameter("start_index", 0)
        self.declare_parameter("max_frames", 0)       # 0 = all
        # Real stereo calibration (full-res *_stereo.yaml). Empty => bundled default.
        self.declare_parameter(
            "left_calib_file",
            _default_calib_path("SAMSON3_SAMSON4_stereo.yaml"),
        )
        self.declare_parameter(
            "right_calib_file",
            _default_calib_path("SAMSON4_SAMSON3_stereo.yaml"),
        )

        top_dir = self.get_parameter("top_dir").value
        bottom_dir = self.get_parameter("bottom_dir").value
        self.left_topic = self.get_parameter("left_image_topic").value
        self.right_topic = self.get_parameter("right_image_topic").value
        self.left_info_topic = self.get_parameter("left_info_topic").value
        self.right_info_topic = self.get_parameter("right_info_topic").value
        self.left_frame = self.get_parameter("left_camera_frame").value
        self.right_frame = self.get_parameter("right_camera_frame").value
        rate_hz = self.get_parameter("rate_hz").value
        self.scale = self.get_parameter("scale").value
        self.loop = self.get_parameter("loop").value
        start_index = self.get_parameter("start_index").value
        max_frames = self.get_parameter("max_frames").value
        left_calib_file = self.get_parameter("left_calib_file").value
        right_calib_file = self.get_parameter("right_calib_file").value

        if not top_dir or not bottom_dir:
            raise RuntimeError(
                "Set 'top_dir' and 'bottom_dir' parameters to the extracted "
                "SAMSON3 / SAMSON4 image folders."
            )

        self.pairs = self._build_pairs(top_dir, bottom_dir)
        if start_index:
            self.pairs = self.pairs[start_index:]
        if max_frames:
            self.pairs = self.pairs[:max_frames]
        if not self.pairs:
            raise RuntimeError("No matched stereo pairs found — check the directories.")

        self.get_logger().info(f"Found {len(self.pairs)} matched stereo pairs")

        self.bridge = CvBridge()
        self.idx = 0

        qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        # camera_info: latched-style so a late-joining stereo_sync still gets it
        info_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_left = self.create_publisher(Image, self.left_topic, qos)
        self.pub_right = self.create_publisher(Image, self.right_topic, qos)
        self.pub_left_info = self.create_publisher(CameraInfo, self.left_info_topic, info_qos)
        self.pub_right_info = self.create_publisher(CameraInfo, self.right_info_topic, info_qos)

        # Per-camera real calibration (None => synthetic). Scaled CameraInfo is built
        # lazily from the first decoded image's size and cached per side.
        self.calib_left = self._load_calib(left_calib_file)
        self.calib_right = self._load_calib(right_calib_file)
        self._info_cache_left = None
        self._info_cache_right = None
        self.get_logger().info(
            "Calibration: "
            f"left={'real' if self.calib_left else 'synthetic'} "
            f"({left_calib_file or 'none'}), "
            f"right={'real' if self.calib_right else 'synthetic'} "
            f"({right_calib_file or 'none'})"
        )

        self.timer = self.create_timer(1.0 / max(rate_hz, 0.01), self._tick)
        self.get_logger().info(
            f"Feeding at {rate_hz} Hz (scale={self.scale}) "
            f"-> {self.left_topic} + {self.right_topic}"
        )

    def _build_pairs(self, top_dir, bottom_dir):
        top = {}
        for p in glob.glob(os.path.join(top_dir, "**", "*.jpg"), recursive=True):
            fid, sec, nsec = _frame_id_and_stamp(p)
            top[fid] = (p, sec, nsec)
        bottom = {}
        for p in glob.glob(os.path.join(bottom_dir, "**", "*.jpg"), recursive=True):
            fid, _, _ = _frame_id_and_stamp(p)
            bottom[fid] = p
        common = sorted(set(top) & set(bottom))
        # (left_path, right_path, sec, nsec) using the TOP camera's timestamp
        return [(top[f][0], bottom[f], top[f][1], top[f][2]) for f in common]

    def _load_calib(self, path):
        """Load a full-res stereo calibration yaml, or return None if no path given."""
        if not path:
            return None
        with open(path) as f:
            c = yaml.safe_load(f)
        w, h = c["imageSize"]
        return {
            "K": np.array(c["cameraMatrix"], dtype=np.float64).reshape(3, 3),
            "D": np.array(c["distCoeffs"], dtype=np.float64).reshape(-1),
            "R": np.array(c["rotation"], dtype=np.float64).reshape(3, 3),
            "P": np.array(c["projectionMatrix"], dtype=np.float64).reshape(3, 4),
            "W": int(w),
            "H": int(h),
        }

    def _build_info_fields(self, calib, width, height):
        """Return (k, d, r, p) flat lists for the published (downscaled) image size."""
        if calib is None:
            # Synthetic fallback: guessed fx, centred principal point, no distortion,
            # identity rectification (stereo_sync passes frames through unchanged).
            fx = fy = 0.9 * width
            cx, cy = width / 2.0, height / 2.0
            k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            d = [0.0, 0.0, 0.0, 0.0, 0.0]
            r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            return k, d, r, p
        # Scale intrinsics from the calibrated full-res size to the published size.
        # K and P scale by S=diag(sx,sy,1); D (dimensionless) and R (rotation) are
        # scale-invariant. Scaling P's whole rows keeps baseline = -P[0,3]/P[0,0].
        sx = width / calib["W"]
        sy = height / calib["H"]
        s = np.diag([sx, sy, 1.0])
        k = (s @ calib["K"]).reshape(-1).tolist()
        p = (s @ calib["P"]).reshape(-1).tolist()
        r = calib["R"].reshape(-1).tolist()
        d = calib["D"].tolist()
        return k, d, r, p

    def _make_info(self, side, width, height, stamp, frame_id):
        if side == "left":
            if self._info_cache_left is None:
                self._info_cache_left = self._build_info_fields(
                    self.calib_left, width, height)
            k, d, r, p = self._info_cache_left
        else:
            if self._info_cache_right is None:
                self._info_cache_right = self._build_info_fields(
                    self.calib_right, width, height)
            k, d, r, p = self._info_cache_right
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = frame_id
        info.width = width
        info.height = height
        info.distortion_model = "plumb_bob"
        info.d = list(d)
        info.k = list(k)
        info.r = list(r)
        info.p = list(p)
        return info

    def _load(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        if self.scale != 1.0:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale,
                             interpolation=cv2.INTER_AREA)
        return img

    def _tick(self):
        if self.idx >= len(self.pairs):
            if self.loop:
                self.idx = 0
            else:
                self.get_logger().info("Finished feeding all pairs.")
                self.timer.cancel()
                return

        left_path, right_path, sec, nsec = self.pairs[self.idx]
        self.idx += 1

        left = self._load(left_path)
        right = self._load(right_path)
        if left is None or right is None:
            self.get_logger().warn(f"Could not read pair {left_path} / {right_path}")
            return

        stamp = rclpy.time.Time(seconds=sec, nanoseconds=nsec).to_msg()
        h, w = left.shape[:2]

        left_msg = self.bridge.cv2_to_imgmsg(left, encoding="bgr8")
        left_msg.header.stamp = stamp
        left_msg.header.frame_id = self.left_frame
        right_msg = self.bridge.cv2_to_imgmsg(right, encoding="bgr8")
        right_msg.header.stamp = stamp
        right_msg.header.frame_id = self.right_frame

        self.pub_left_info.publish(self._make_info("left", w, h, stamp, self.left_frame))
        self.pub_right_info.publish(self._make_info("right", w, h, stamp, self.right_frame))
        self.pub_left.publish(left_msg)
        self.pub_right.publish(right_msg)

        self.get_logger().info(
            f"[{self.idx}/{len(self.pairs)}] published pair ({w}x{h})",
            throttle_duration_sec=2.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = BagImageFeederNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
