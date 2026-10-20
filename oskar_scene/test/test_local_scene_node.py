from unittest.mock import patch

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, PointStamped, TransformStamped
from moveit_msgs.msg import PlanningScene
from oskar_msgs.msg import ThinningCandidate
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger

from oskar_scene.filters import carve_target, clip_range, crop_roi, voxel_downsample
from oskar_scene.local_scene_node import (
    LocalSceneNode, do_transform_point, transform_cloud_points,
)
from oskar_scene.point_cloud import cloud_to_points, points_to_cloud


def test_octomap_readiness_uses_pose_header_and_checks_payload():
    rclpy.init()
    node = LocalSceneNode()
    try:
        scene = PlanningScene()
        map_with_pose = scene.world.octomap
        octomap = map_with_pose.octomap
        map_with_pose.header.frame_id = 'base_link'
        octomap.resolution = 0.02
        assert not node._has_expected_octomap(scene)
        octomap.data = [1]
        assert node._has_expected_octomap(scene)
        map_with_pose.header.frame_id = ''
        octomap.header.frame_id = 'base_link'
        assert not node._has_expected_octomap(scene)
        map_with_pose.header.frame_id = 'wrong_frame'
        assert not node._has_expected_octomap(scene)
        map_with_pose.header.frame_id = 'base_link'
        octomap.resolution = 0.05
        assert not node._has_expected_octomap(scene)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_range_rejection_precedes_transform_and_preserves_sensor_header():
    rclpy.init()
    node = LocalSceneNode()
    try:
        transform = TransformStamped()
        transform.header.frame_id = 'base_link'
        transform.child_frame_id = 'fixture_sensor'
        transform.transform.translation.x = 10.0
        transform.transform.rotation.z = np.sqrt(0.5)
        transform.transform.rotation.w = np.sqrt(0.5)
        node._tf_buffer.set_transform_static(transform, 'test')
        stamp = Time(sec=42, nanosec=123)
        cloud = points_to_cloud(
            [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            'fixture_sensor', stamp)
        target = ThinningCandidate(diameter=0.1)

        with patch('oskar_scene.local_scene_node.transform_cloud_points',
               wraps=transform_cloud_points) as point_transform:
            points, frame, result_stamp = node._filter_view(
                cloud, np.array([10.0, 1.5, 0.0]), target)

        np.testing.assert_allclose(points, [[1.0, 0.0, 0.0]], atol=1e-12)
        assert point_transform.call_count == 2
        assert point_transform.call_args_list[0].args[0].shape == (1, 3)
        assert frame == 'fixture_sensor'
        assert result_stamp == stamp
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_bulk_filter_matches_pointwise_tf_and_handles_empty_cloud():
    rclpy.init()
    node = LocalSceneNode()
    try:
        transform = TransformStamped()
        transform.header.frame_id = 'base_link'
        transform.child_frame_id = 'fixture_sensor'
        transform.transform.translation.x = 0.3
        transform.transform.translation.y = -0.2
        transform.transform.translation.z = 0.1
        transform.transform.rotation.x = 0.2
        transform.transform.rotation.y = -0.3
        transform.transform.rotation.z = 0.4
        transform.transform.rotation.w = np.sqrt(0.71)
        node._tf_buffer.set_transform_static(transform, 'test')
        stamp = Time(sec=42, nanosec=123)
        source = np.random.default_rng(42).uniform(-1.5, 1.5, (300, 3))
        cloud = points_to_cloud(source, 'fixture_sensor', stamp)
        target_point = np.array([0.0, 0.0, 1.0])
        target = ThinningCandidate(diameter=0.1)

        source = clip_range(cloud_to_points(cloud), np.zeros(3), 0.3, 2.5)
        expected = []
        for coordinates in source:
            transformed = do_transform_point(PointStamped(point=Point(
                x=float(coordinates[0]), y=float(coordinates[1]),
                z=float(coordinates[2]))), transform)
            expected.append([transformed.point.x, transformed.point.y, transformed.point.z])
        expected = crop_roi(np.array(expected), target_point, [2.0, 2.0, 2.0])
        expected = carve_target(expected, target_point, target.diameter, 0.02)
        expected = voxel_downsample(expected, 0.01)

        result, frame, result_stamp = node._filter_view(cloud, target_point, target)
        np.testing.assert_allclose(
            transform_cloud_points(result, transform.transform), expected, atol=1e-12)
        assert len(result) > 0
        assert frame == cloud.header.frame_id
        assert result_stamp == stamp

        empty = points_to_cloud(np.empty((0, 3)), 'fixture_sensor', stamp)
        result, frame, result_stamp = node._filter_view(empty, target_point, target)
        assert result.shape == (0, 3)
        assert frame == 'fixture_sensor'
        assert result_stamp == stamp
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_capture_limit_and_incomplete_build_clear_readiness():
    rclpy.init()
    node = LocalSceneNode()
    try:
        node._call_empty = lambda _client: True
        assert node._begin_capture(Trigger.Request(), Trigger.Response()).success

        for _ in range(4):
            node._on_cloud(PointCloud2())
        assert len(node._arrival_views) == 3
        assert not node._capture_active

        assert node._begin_capture(Trigger.Request(), Trigger.Response()).success
        assert len(node._arrival_views) == 0
        node._set_ready(True)
        node._on_cloud(PointCloud2())
        response = node._build(Trigger.Request(), Trigger.Response())
        assert not response.success
        assert 'incomplete' in response.message
        assert not node._ready
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_build_fails_closed_when_moveit_clear_is_unavailable():
    rclpy.init()
    node = LocalSceneNode()
    try:
        class UnavailableService:
            srv_name = '/clear_octomap'

            def wait_for_service(self, timeout_sec):
                return False

        node._arrival_views = [PointCloud2() for _ in range(node._view_limit())]
        node._target = ThinningCandidate()
        node._target.pose.header.frame_id = 'base_link'
        node._clear_octomap = UnavailableService()
        with patch.object(node, '_target_point', return_value=np.zeros(3)), \
                patch.object(node, '_filter_view', return_value=(
                    np.array([[1.0, 0.0, 0.0]]), 'fixture_sensor', Time(sec=1))):
            response = node._build(Trigger.Request(), Trigger.Response())
        assert not response.success and response.message == 'MoveIt /clear_octomap is unavailable'
        assert not node._ready
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_build_fails_closed_when_moveit_scene_is_unavailable():
    rclpy.init()
    node = LocalSceneNode()
    try:
        class UnavailableService:
            srv_name = '/get_planning_scene'

            def wait_for_service(self, timeout_sec):
                return False

        node._arrival_views = [PointCloud2() for _ in range(node._view_limit())]
        node._target = ThinningCandidate()
        node._target.pose.header.frame_id = 'base_link'
        node._call_empty = lambda _client: True
        node._get_scene = UnavailableService()
        node._wait_for_updater = lambda: True
        node._tf_timeout = lambda: 0.01
        node._scene_update_timeout = lambda: 0.05
        with patch.object(node, '_target_point', return_value=np.zeros(3)), \
                patch.object(node, '_filter_view', return_value=(
                    np.array([[1.0, 0.0, 0.0]]), 'fixture_sensor', Time(sec=1))):
            response = node._build(Trigger.Request(), Trigger.Response())
        assert not response.success and response.message == 'MoveIt octomap is empty or not ready'
        assert not node._ready
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_build_fails_closed_when_updater_subscriber_is_missing():
    rclpy.init()
    node = LocalSceneNode()
    try:
        node._arrival_views = [PointCloud2() for _ in range(node._view_limit())]
        node._target = ThinningCandidate()
        node._target.pose.header.frame_id = 'base_link'
        node._call_empty = lambda _client: True
        node._wait_for_updater = lambda: False
        with patch.object(node, '_target_point', return_value=np.zeros(3)), \
                patch.object(node, '_filter_view', return_value=(
                    np.array([[1.0, 0.0, 0.0]]), 'fixture_sensor', Time(sec=1))):
            response = node._build(Trigger.Request(), Trigger.Response())
        assert not response.success
        assert response.message == 'MoveIt octomap updater has no cloud subscriber'
        assert not node._ready
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_build_fails_closed_when_octomap_update_times_out():
    rclpy.init()
    node = LocalSceneNode()
    try:
        requests = []
        node._arrival_views = [PointCloud2() for _ in range(node._view_limit())]
        node._target = ThinningCandidate()
        node._target.pose.header.frame_id = 'base_link'
        node._call_empty = lambda _client: True
        node._wait_for_updater = lambda: True
        node._tf_timeout = lambda: 0.01
        node._scene_update_timeout = lambda: 0.05
        node._request_planning_scene = lambda: requests.append(True) or PlanningScene()
        with patch.object(node, '_target_point', return_value=np.zeros(3)), \
                patch.object(node, '_filter_view', return_value=(
                    np.array([[1.0, 0.0, 0.0]]), 'fixture_sensor', Time(sec=1))):
            response = node._build(Trigger.Request(), Trigger.Response())
        assert not response.success
        assert response.message == 'MoveIt octomap is empty or not ready'
        assert requests and not node._ready
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_target_snapshot_and_clear_reset_cached_views():
    rclpy.init()
    node = LocalSceneNode()
    try:
        node._call_empty = lambda _client: True
        assert not node._arm_snapshot(Trigger.Request(), Trigger.Response()).success
        node._begin_capture(Trigger.Request(), Trigger.Response())
        for _ in range(3):
            node._on_cloud(PointCloud2())
        assert not node._build(Trigger.Request(), Trigger.Response()).success

        target = ThinningCandidate()
        target.pose.header.frame_id = 'base_link'
        node._on_target(target)
        assert node._arm_snapshot(Trigger.Request(), Trigger.Response()).success
        snapshot = PointCloud2()
        snapshot.header.frame_id = 'fixture_sensor'
        node._on_cloud(snapshot)
        node._on_cloud(PointCloud2())
        assert node._snapshot.header.frame_id == 'fixture_sensor'
        assert len(node._arrival_views) == 3

        node._set_ready(True)
        node._on_target(target)
        assert not node._ready
        assert node._clear(Trigger.Request(), Trigger.Response()).success
        assert not node._arrival_views
        assert node._snapshot is None
        assert node._target is None
        assert not node._ready
    finally:
        node.destroy_node()
        rclpy.shutdown()