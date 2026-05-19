#!/usr/bin/env python3
"""
Stereo synchronization and rectification node.

Camera info is received once via independent subscriptions (it is essentially
static).  Only the two image topics are time-synchronised, which is far more
robust than a 4-topic sync when camera_info timestamps don't perfectly align
with image timestamps (common in Isaac Sim).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from message_filters import ApproximateTimeSynchronizer, Subscriber
import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from oskar_msgs.msg import StereoPair


class StereoSyncNode(Node):
    def __init__(self):
        super().__init__('stereo_sync_node')

        self.declare_parameter('left_image_topic',  '/rgb_static_1')
        self.declare_parameter('right_image_topic', '/rgb_static_2')
        self.declare_parameter('left_info_topic',   '/static_camera_info_1')
        self.declare_parameter('right_info_topic',  '/static_camera_info_2')
        self.declare_parameter('left_camera_frame', 'static_camera_1')
        self.declare_parameter('right_camera_frame','static_camera_2')
        self.declare_parameter('sync_slop',  0.05)
        self.declare_parameter('queue_size', 10)

        left_img_topic   = self.get_parameter('left_image_topic').value
        right_img_topic  = self.get_parameter('right_image_topic').value
        left_info_topic  = self.get_parameter('left_info_topic').value
        right_info_topic = self.get_parameter('right_info_topic').value
        self.left_camera_frame = self.get_parameter('left_camera_frame').value
        sync_slop  = self.get_parameter('sync_slop').value
        queue_size = self.get_parameter('queue_size').value

        self.bridge = CvBridge()
        self._left_info  = None
        self._right_info = None
        self._maps_ready = False
        self._map_lx = self._map_ly = self._map_rx = self._map_ry = None

        # Camera info: subscribe once with RELIABLE qos; cache on first message.
        reliable_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                  durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(CameraInfo, left_info_topic,
                                 self._left_info_cb,  reliable_qos)
        self.create_subscription(CameraInfo, right_info_topic,
                                 self._right_info_cb, reliable_qos)

        # Images: sync only these two topics.
        img_qos = QoSProfile(depth=queue_size, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE)
        sub_l = Subscriber(self, Image, left_img_topic,  qos_profile=img_qos)
        sub_r = Subscriber(self, Image, right_img_topic, qos_profile=img_qos)
        self.ts = ApproximateTimeSynchronizer([sub_l, sub_r],
                                              queue_size=queue_size, slop=sync_slop)
        self.ts.registerCallback(self._img_callback)

        self.pub = self.create_publisher(StereoPair, '/stereo/sync_pair', queue_size)

        self.get_logger().info(
            f'stereo_sync_node ready — syncing {left_img_topic} + {right_img_topic}'
        )

    # ------------------------------------------------------------------
    # Camera info callbacks (fire once, then ignored)
    # ------------------------------------------------------------------

    def _left_info_cb(self, msg: CameraInfo):
        if self._left_info is None:
            self._left_info = msg
            self.get_logger().info('Left camera info received')
            self._try_init_rectification()

    def _right_info_cb(self, msg: CameraInfo):
        if self._right_info is None:
            self._right_info = msg
            self.get_logger().info('Right camera info received')
            self._try_init_rectification()

    def _try_init_rectification(self):
        if self._left_info is None or self._right_info is None:
            return

        li, ri = self._left_info, self._right_info
        K_l = np.array(li.k, dtype=np.float64).reshape(3, 3)
        D_l = np.array(li.d, dtype=np.float64)
        R_l = np.array(li.r, dtype=np.float64).reshape(3, 3)
        P_l = np.array(li.p, dtype=np.float64).reshape(3, 4)[:, :3]

        K_r = np.array(ri.k, dtype=np.float64).reshape(3, 3)
        D_r = np.array(ri.d, dtype=np.float64)
        R_r = np.array(ri.r, dtype=np.float64).reshape(3, 3)
        P_r = np.array(ri.p, dtype=np.float64).reshape(3, 4)[:, :3]

        size = (li.width, li.height)

        already_rectified = (np.allclose(D_l, 0) and np.allclose(R_l, np.eye(3)) and
                             np.allclose(D_r, 0) and np.allclose(R_r, np.eye(3)))
        if already_rectified:
            self.get_logger().info('Images already rectified — skipping remap')
        else:
            self._map_lx, self._map_ly = cv2.initUndistortRectifyMap(
                K_l, D_l, R_l, P_l, size, cv2.CV_32F)
            self._map_rx, self._map_ry = cv2.initUndistortRectifyMap(
                K_r, D_r, R_r, P_r, size, cv2.CV_32F)
            self.get_logger().info('Rectification maps built')

        self._maps_ready = True

    # ------------------------------------------------------------------
    # Image sync callback
    # ------------------------------------------------------------------

    def _img_callback(self, left_msg: Image, right_msg: Image):
        if not self._maps_ready:
            self.get_logger().warn(
                'Image pair received but camera info not yet available — dropping frame',
                throttle_duration_sec=2.0,
            )
            return

        try:
            left  = self.bridge.imgmsg_to_cv2(left_msg,  desired_encoding='bgr8')
            right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')

            if self._map_lx is not None:
                left  = cv2.remap(left,  self._map_lx, self._map_ly, cv2.INTER_LINEAR)
                right = cv2.remap(right, self._map_rx, self._map_ry, cv2.INTER_LINEAR)

            pair = StereoPair()
            pair.header.stamp    = left_msg.header.stamp
            pair.header.frame_id = self.left_camera_frame
            pair.left_image      = self.bridge.cv2_to_imgmsg(left,  encoding='bgr8')
            pair.right_image     = self.bridge.cv2_to_imgmsg(right, encoding='bgr8')
            pair.left_image.header  = left_msg.header
            pair.right_image.header = right_msg.header
            pair.left_info  = self._left_info
            pair.right_info = self._right_info

            self.pub.publish(pair)

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = StereoSyncNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
