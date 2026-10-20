"""Build a target-carved local octomap from captured PointCloud2 views."""

from __future__ import annotations

import copy
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
from oskar_msgs.msg import ThinningCandidate
from rclpy.duration import Duration
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, UInt32
from std_srvs.srv import Empty, Trigger
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_sensor_msgs.tf2_sensor_msgs import transform_points as transform_cloud_points

from oskar_scene.filters import (
    carve_corridor, carve_target, clip_range, crop_roi, voxel_downsample)
from oskar_scene.point_cloud import cloud_to_points, points_to_cloud


class LocalSceneNode(Node):
    """Capture a bounded view set and publish filtered views for MoveIt."""

    def __init__(self):
        super().__init__('local_scene_node')
        self.declare_parameter('planning_frame', 'base_link')
        self.declare_parameter('roi_size_xyz', [2.0, 2.0, 2.0])
        self.declare_parameter('carve_margin', 0.02)
        # Cylinder along the target's +Z (its approach axis) cleared as well, so
        # the linear approach is not blocked by the foliage in front of the
        # blossom. Length 0 carves the sphere only.
        self.declare_parameter('corridor_length', 0.0)
        self.declare_parameter('corridor_radius', 0.05)
        self.declare_parameter('voxel_leaf_size', 0.01)
        self.declare_parameter('min_range', 0.30)
        self.declare_parameter('max_range', 2.50)
        self.declare_parameter('accumulate_views', 3)
        self.declare_parameter('clear_before_capture', True)
        self.declare_parameter('tf_timeout', 1.0)
        self.declare_parameter('scene_update_timeout', 5.0)
        self.declare_parameter('view_publish_interval', 0.10)
        self.declare_parameter('octomap_frame', 'base_link')
        self.declare_parameter('octomap_resolution', 0.02)

        self._lock = threading.RLock()
        self._callbacks = ReentrantCallbackGroup()
        self._arrival_views = []
        self._snapshot = None
        self._target = None
        self._capture_active = False
        self._snapshot_armed = False
        self._ready = False
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)

        self._filtered_pub = self.create_publisher(PointCloud2, 'filtered_cloud', 10)
        self._arrival_ready_pub = self.create_publisher(Bool, 'arrival_ready', 1)
        self._ready_pub = self.create_publisher(Bool, 'ready', 1)
        self._view_count_pub = self.create_publisher(UInt32, 'view_count', 1)
        self.create_subscription(
            PointCloud2, 'input_cloud', self._on_cloud, 10,
            callback_group=self._callbacks)
        self.create_subscription(
            ThinningCandidate, 'target', self._on_target, 10,
            callback_group=self._callbacks)
        self.create_service(
            Trigger, 'begin_arrival_capture', self._begin_capture,
            callback_group=self._callbacks)
        self.create_service(
            Trigger, 'arm_blossom_snapshot', self._arm_snapshot,
            callback_group=self._callbacks)
        self.create_service(
            Trigger, 'build', self._build, callback_group=self._callbacks)
        self.create_service(
            Trigger, 'clear', self._clear, callback_group=self._callbacks)
        self._clear_octomap = self.create_client(Empty, '/clear_octomap')
        self._get_scene = self.create_client(GetPlanningScene, '/get_planning_scene')
        self._publish_status()

    def _on_cloud(self, cloud):
        with self._lock:
            if self._snapshot_armed:
                self._snapshot = cloud
                self._snapshot_armed = False
                return
            if self._capture_active and len(self._arrival_views) < self._view_limit():
                self._arrival_views.append(cloud)
                if len(self._arrival_views) == self._view_limit():
                    self._capture_active = False
                self._publish_status()

    def _on_target(self, target):
        with self._lock:
            self._target = copy.deepcopy(target)
            self._set_ready(False)

    def _begin_capture(self, _request, response):
        with self._lock:
            self._arrival_views.clear()
            self._snapshot = None
            self._snapshot_armed = False
            self._capture_active = True
            self._set_ready(False)
        if self.get_parameter('clear_before_capture').value:
            self._call_empty(self._clear_octomap)
        response.success = True
        response.message = f'capturing {self._view_limit()} arrival views'
        self._publish_status()
        return response

    def _arm_snapshot(self, _request, response):
        with self._lock:
            if self._target is None:
                response.success = False
                response.message = 'publish a target ThinningCandidate first'
                return response
            self._snapshot_armed = True
        response.success = True
        response.message = 'waiting for one blossom snapshot'
        return response

    def _clear(self, _request, response):
        with self._lock:
            self._arrival_views.clear()
            self._snapshot = None
            self._target = None
            self._capture_active = False
            self._snapshot_armed = False
            self._set_ready(False)
        self._call_empty(self._clear_octomap)
        response.success = True
        response.message = 'scene and captured views cleared'
        self._publish_status()
        return response

    def _build(self, _request, response):
        with self._lock:
            self._set_ready(False)
            views = list(self._arrival_views)
            if self._snapshot is not None:
                views.append(self._snapshot)
            target = copy.deepcopy(self._target)
        if len(self._arrival_views) < self._view_limit():
            response.success = False
            response.message = 'arrival capture is incomplete'
            return response
        if target is None or not target.pose.header.frame_id:
            response.success = False
            response.message = 'target ThinningCandidate with a frame is required'
            return response
        try:
            target_point = self._target_point(target)
            transformed_views = [self._filter_view(cloud, target_point, target) for cloud in views]
            transformed_views = [view for view in transformed_views if len(view[0])]
            if not transformed_views:
                raise ValueError('all points were removed by the scene filters')
            if not self._call_empty(self._clear_octomap):
                raise RuntimeError('MoveIt /clear_octomap is unavailable')
            if not self._wait_for_updater():
                raise RuntimeError('MoveIt octomap updater has no cloud subscriber')
            for points, frame_id, stamp in transformed_views:
                self._filtered_pub.publish(points_to_cloud(points, frame_id, stamp))
                time.sleep(float(self.get_parameter('view_publish_interval').value))
            if not self._wait_for_octomap():
                raise RuntimeError('MoveIt octomap is empty or not ready')
            self._set_ready(True)
            response.success = True
            response.message = f'published {len(transformed_views)} filtered views'
        except (RuntimeError, TransformException, ValueError) as error:
            self.get_logger().error(str(error))
            self._set_ready(False)
            response.success = False
            response.message = str(error)
        return response

    def _filter_view(self, cloud, target_point, target):
        points = cloud_to_points(cloud)
        points = clip_range(points, np.zeros(3), self.get_parameter('min_range').value,
                            self.get_parameter('max_range').value)
        transform = self._tf_buffer.lookup_transform(
            self._planning_frame(), cloud.header.frame_id,
            Time.from_msg(cloud.header.stamp),
            timeout=Duration(seconds=self._tf_timeout()))
        transformed = transform_cloud_points(points, transform.transform)
        transformed = crop_roi(transformed, target_point, self.get_parameter('roi_size_xyz').value)
        transformed = carve_target(transformed, target_point, target.diameter,
                                   self.get_parameter('carve_margin').value)
        transformed = carve_corridor(
            transformed, target_point, self._target_axis(target),
            float(self.get_parameter('corridor_length').value),
            float(self.get_parameter('corridor_radius').value))
        transformed = voxel_downsample(transformed, self.get_parameter('voxel_leaf_size').value)
        # MoveIt needs the original sensor frame/stamp to ray-trace free space correctly.
        inverse = self._tf_buffer.lookup_transform(
            cloud.header.frame_id, self._planning_frame(),
            Time.from_msg(cloud.header.stamp),
            timeout=Duration(seconds=self._tf_timeout()))
        restored = transform_cloud_points(transformed, inverse.transform)
        return restored, cloud.header.frame_id, cloud.header.stamp

    def _target_point(self, target):
        pose = target.pose
        transform = self._tf_buffer.lookup_transform(
            self._planning_frame(), pose.header.frame_id,
            Time.from_msg(pose.header.stamp),
            timeout=Duration(seconds=self._tf_timeout()))
        point = do_transform_point(PointStamped(
            point=pose.pose.position), transform)
        return np.array([point.point.x, point.point.y, point.point.z], dtype=np.float64)

    def _target_axis(self, target):
        """The target's +Z in the planning frame; zero if it cannot be had."""
        if float(self.get_parameter('corridor_length').value) <= 0.0:
            return np.zeros(3)
        q = target.pose.pose.orientation
        x, y, z, w = q.x, q.y, q.z, q.w
        axis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        transform = self._tf_buffer.lookup_transform(
            self._planning_frame(), target.pose.header.frame_id,
            Time.from_msg(target.pose.header.stamp),
            timeout=Duration(seconds=self._tf_timeout()))
        r = transform.transform.rotation
        x, y, z, w = r.x, r.y, r.z, r.w
        rotation = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        return rotation @ axis

    def _call_empty(self, client):
        if not client.wait_for_service(timeout_sec=self._tf_timeout()):
            self.get_logger().warning(f'service unavailable: {client.srv_name}')
            return False
        future = client.call_async(Empty.Request())
        deadline = time.monotonic() + self._tf_timeout()
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            self.get_logger().warning(f'service timed out: {client.srv_name}')
            return False
        return future.exception() is None

    def _wait_for_updater(self):
        deadline = time.monotonic() + self._scene_update_timeout()
        while time.monotonic() < deadline:
            if self._filtered_pub.get_subscription_count() > 0:
                return True
            time.sleep(0.01)
        return False

    def _wait_for_octomap(self):
        deadline = time.monotonic() + self._scene_update_timeout()
        while time.monotonic() < deadline:
            scene = self._request_planning_scene()
            if scene is not None and self._has_expected_octomap(scene):
                return True
            time.sleep(0.05)
        return False

    def _request_planning_scene(self):
        if not self._get_scene.wait_for_service(timeout_sec=self._tf_timeout()):
            return None
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.OCTOMAP
        future = self._get_scene.call_async(request)
        deadline = time.monotonic() + self._tf_timeout()
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done() or future.exception() is not None:
            return None
        return future.result().scene

    def _has_expected_octomap(self, scene):
        map_with_pose = scene.world.octomap
        octomap = map_with_pose.octomap
        return bool(octomap.data) and (
            map_with_pose.header.frame_id == self.get_parameter('octomap_frame').value
            and abs(octomap.resolution - self.get_parameter('octomap_resolution').value) < 1e-9)

    def _set_ready(self, value):
        self._ready = value
        message = Bool(data=value)
        self._ready_pub.publish(message)

    def _publish_status(self):
        self._arrival_ready_pub.publish(Bool(data=len(self._arrival_views) == self._view_limit()))
        self._view_count_pub.publish(UInt32(data=len(self._arrival_views)))
        self._ready_pub.publish(Bool(data=self._ready))

    def _view_limit(self):
        return int(self.get_parameter('accumulate_views').value)

    def _planning_frame(self):
        return str(self.get_parameter('planning_frame').value)

    def _tf_timeout(self):
        return float(self.get_parameter('tf_timeout').value)

    def _scene_update_timeout(self):
        return float(self.get_parameter('scene_update_timeout').value)


def main(args=None):
    rclpy.init(args=args)
    node = LocalSceneNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
