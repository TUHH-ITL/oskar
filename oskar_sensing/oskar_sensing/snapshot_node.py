"""Republish one fresh PointCloud2 snapshot on demand."""

from __future__ import annotations

import threading
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger


class SnapshotNode(Node):
    """Bridge live, SVO, or bag clouds behind a trigger-based snapshot API."""

    def __init__(self):
        super().__init__('snapshot_node')
        self.declare_parameter('source_cloud_topic', '/zed/zed_node/point_cloud/cloud_registered')
        self.declare_parameter('output_cloud_topic', '/scene/input_cloud')
        self.declare_parameter('snapshot_timeout', 5.0)
        self._lock = threading.Lock()
        self._armed = False
        self._pending = None
        self._publisher = self.create_publisher(
            PointCloud2, self.get_parameter('output_cloud_topic').value, 10)
        # The service waits for the next cloud, so both callbacks must be able
        # to run at once.
        callbacks = ReentrantCallbackGroup()
        self.create_subscription(
            PointCloud2, self.get_parameter('source_cloud_topic').value,
            self._on_cloud, 10, callback_group=callbacks)
        self.create_service(Trigger, 'capture_snapshot', self._capture,
                            callback_group=callbacks)

    def _on_cloud(self, cloud):
        with self._lock:
            if self._armed:
                self._pending = (cloud, time.monotonic())
                self._armed = False

    def _capture(self, _request, response):
        deadline = time.monotonic() + float(self.get_parameter('snapshot_timeout').value)
        with self._lock:
            self._pending = None
            self._armed = True
        while time.monotonic() < deadline and rclpy.ok():
            with self._lock:
                pending = self._pending
                self._pending = None
            if pending is not None:
                self._publisher.publish(pending[0])
                response.success = True
                response.message = (
                    f'published snapshot frame={pending[0].header.frame_id} '
                    f'stamp={pending[0].header.stamp.sec}.{pending[0].header.stamp.nanosec:09d}')
                return response
            time.sleep(0.01)
        with self._lock:
            self._armed = False
        response.success = False
        response.message = 'snapshot timed out waiting for a complete PointCloud2'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = SnapshotNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
