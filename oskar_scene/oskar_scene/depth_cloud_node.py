"""Turn a depth image into the PointCloud2 the local scene consumes.

The simulated wrist camera publishes depth and camera_info but no cloud; the
real ZED publishes a registered cloud, which goes through oskar_sensing instead.
Publishes only while someone listens, at most max_rate_hz.
"""

from __future__ import annotations

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

from oskar_scene.filters import depth_to_points
from oskar_scene.point_cloud import points_to_cloud


class DepthCloudNode(Node):
    def __init__(self):
        super().__init__('depth_cloud_node')
        self.declare_parameter('depth_topic', '/wrist_cam/left/depth')
        self.declare_parameter('camera_info_topic', '/wrist_cam/left/camera_info')
        self.declare_parameter('cloud_topic', '/scene/input_cloud')
        self.declare_parameter('stride', 4)
        self.declare_parameter('max_range', 2.5)
        self.declare_parameter('max_rate_hz', 5.0)

        self._bridge = CvBridge()
        self._info = None
        self._last = None
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(
            PointCloud2, self.get_parameter('cloud_topic').value, qos)
        self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value, self._on_info, qos)
        self.create_subscription(
            Image, self.get_parameter('depth_topic').value, self._on_depth, qos)

    def _on_info(self, msg):
        self._info = msg

    def _on_depth(self, msg):
        if self._info is None or not self._pub.get_subscription_count():
            return
        now = self.get_clock().now()
        period = 1.0 / float(self.get_parameter('max_rate_hz').value)
        if self._last is not None and (now - self._last).nanoseconds * 1e-9 < period:
            return
        self._last = now
        depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if msg.encoding == '16UC1':
            depth = depth.astype(np.float64) / 1000.0
        k = self._info.k
        points = depth_to_points(depth, k[0], k[4], k[2], k[5],
                                 int(self.get_parameter('stride').value))
        points = points[points[:, 2] <= float(self.get_parameter('max_range').value)]
        self._pub.publish(points_to_cloud(points, msg.header.frame_id, msg.header.stamp))


def main(args=None):
    rclpy.init(args=args)
    node = DepthCloudNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
