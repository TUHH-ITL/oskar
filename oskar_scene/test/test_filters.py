import numpy as np
import pytest

from oskar_scene.filters import (
    carve_target,
    clip_range,
    crop_roi,
    transform_points,
    voxel_downsample,
)


def test_transform_and_range_use_sensor_origin():
    points = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    transformed = transform_points(points, np.eye(3), [1.0, 0.0, 0.0])
    result = clip_range(transformed, [1.0, 0.0, 0.0], 0.5, 1.5)
    np.testing.assert_allclose(result, [[2.0, 0.0, 0.0]])


def test_roi_and_carve_boundaries_are_exclusive_for_removed_points():
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.11, 0.0, 0.0],
        [0.99, 0.0, 0.0],
    ])
    roi = crop_roi(points, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
    result = carve_target(roi, [0.0, 0.0, 0.0], 0.2, 0.01)
    np.testing.assert_allclose(result, [[0.99, 0.0, 0.0]])


def test_voxel_downsample_is_deterministic_and_keeps_source_points():
    points = np.array([[0.01, 0.01, 0.01], [0.09, 0.09, 0.09], [0.21, 0.0, 0.0]])
    result = voxel_downsample(points, 0.1)
    np.testing.assert_allclose(result, [[0.09, 0.09, 0.09], [0.21, 0.0, 0.0]])


def test_voxel_downsample_rejects_invalid_leaf():
    try:
        voxel_downsample(np.zeros((1, 3)), 0.0)
    except ValueError as error:
        assert 'positive' in str(error)
    else:
        raise AssertionError('expected ValueError')


@pytest.mark.parametrize('points, leaf_size', [
    (np.random.default_rng(42).uniform(-1.0, 1.0, (1500, 3)), 0.2),
    (np.array([[0.75, 0.5, 0.5], [0.25, 0.5, 0.5],
               [0.75, 0.5, 0.5], [-0.25, -0.5, -0.5],
               [-0.75, -0.5, -0.5], [np.nan, 0.0, 0.0]]), 1.0),
    (np.empty((0, 3)), 0.1),
])
def test_voxel_matches_reference_selection_order_and_ties(points, leaf_size):
    finite = points[np.all(np.isfinite(points), axis=1)]
    keys = np.floor(finite / leaf_size).astype(np.int64)
    expected = []
    for key in np.unique(keys, axis=0):
        members = finite[np.all(keys == key, axis=1)]
        center = (key.astype(np.float64) + 0.5) * leaf_size
        distances = np.sum((members - center) ** 2, axis=1)
        expected.append(members[np.argmin(distances)])

    np.testing.assert_array_equal(
        voxel_downsample(points, leaf_size), np.asarray(expected).reshape(-1, 3))


def test_voxel_keeps_roi_and_carve_masks():
    points = np.random.default_rng(7).uniform(-1.0, 1.0, (1000, 3))
    cropped = crop_roi(points, np.zeros(3), [1.0, 1.0, 1.0])
    carved = carve_target(cropped, np.zeros(3), 0.4, 0.02)
    result = voxel_downsample(carved, 0.5)

    assert len(result) > 0
    assert np.all(np.abs(result) <= 0.5)
    assert np.all(np.linalg.norm(result, axis=1) > 0.22)
    assert all(np.any(np.all(carved == point, axis=1)) for point in result)


def test_corridor_removes_only_the_cylinder_in_front_of_the_target():
    from oskar_scene.filters import carve_corridor
    points = np.array([
        [0.0, 0.05, 0.0],    # inside, along the axis
        [0.03, 0.10, 0.0],   # inside, off axis within the radius
        [0.08, 0.10, 0.0],   # outside the radius
        [0.0, 0.20, 0.0],    # beyond the length
        [0.0, -0.05, 0.0],   # behind the target
    ])
    result = carve_corridor(points, [0, 0, 0], [0, 1, 0], 0.15, 0.05)
    np.testing.assert_allclose(result, points[2:])
    np.testing.assert_allclose(carve_corridor(points, [0, 0, 0], [0, 1, 0], 0.0, 0.05), points)


def test_depth_back_projection_matches_the_pinhole_model():
    from oskar_scene.filters import depth_to_points
    depth = np.full((4, 6), 2.0)
    depth[0, 0] = np.nan
    depth[1, 1] = 0.0
    points = depth_to_points(depth, fx=100.0, fy=100.0, cx=3.0, cy=2.0)
    assert len(points) == 22
    np.testing.assert_allclose(points[-1], [(5 - 3) * 2 / 100, (3 - 2) * 2 / 100, 2.0])
    assert len(depth_to_points(depth, 100.0, 100.0, 3.0, 2.0, stride=2)) == 5
