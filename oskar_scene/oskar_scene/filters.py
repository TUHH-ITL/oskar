"""Pure NumPy filters used to prepare local collision clouds."""

from __future__ import annotations

import numpy as np


def transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply a 3x3 rotation and 3-vector translation to XYZ points."""
    points = _points(points)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    return points @ rotation.T + translation


def clip_range(
    points: np.ndarray,
    sensor_origin: np.ndarray,
    min_range: float,
    max_range: float,
) -> np.ndarray:
    """Keep finite points whose distance from the sensor lies in the range."""
    points = _points(points)
    origin = np.asarray(sensor_origin, dtype=np.float64).reshape(3)
    distances = np.linalg.norm(points - origin, axis=1)
    return points[(distances >= min_range) & (distances <= max_range)]


def crop_roi(points: np.ndarray, center: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Keep points inside an axis-aligned box centered at *center*."""
    points = _points(points)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    half_size = np.asarray(size, dtype=np.float64).reshape(3) / 2.0
    return points[np.all(np.abs(points - center) <= half_size, axis=1)]


def carve_target(
    points: np.ndarray,
    center: np.ndarray,
    diameter: float,
    margin: float,
) -> np.ndarray:
    """Remove points at or inside the target blossom's carve sphere."""
    points = _points(points)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    radius = max(0.0, float(diameter) / 2.0 + float(margin))
    distances = np.linalg.norm(points - center, axis=1)
    return points[distances > radius]


def carve_corridor(
    points: np.ndarray,
    start: np.ndarray,
    axis: np.ndarray,
    length: float,
    radius: float,
) -> np.ndarray:
    """Remove points inside the cylinder from *start* along *axis* for *length*."""
    points = _points(points)
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(axis)
    if length <= 0.0 or radius <= 0.0 or norm < 1e-12:
        return points
    axis = axis / norm
    relative = points - np.asarray(start, dtype=np.float64).reshape(3)
    along = relative @ axis
    across = np.linalg.norm(relative - np.outer(along, axis), axis=1)
    inside = (along >= 0.0) & (along <= length) & (across <= radius)
    return points[~inside]


def depth_to_points(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int = 1,
) -> np.ndarray:
    """Back-project a metric depth image into optical-frame XYZ, every stride-th pixel."""
    depth = np.asarray(depth, dtype=np.float64)[::stride, ::stride]
    rows, cols = np.indices(depth.shape)
    u = cols * stride
    v = rows * stride
    points = np.stack([(u - cx) * depth / fx, (v - cy) * depth / fy, depth], axis=-1)
    points = points.reshape(-1, 3)
    return points[np.all(np.isfinite(points), axis=1) & (points[:, 2] > 0.0)]


def voxel_downsample(points: np.ndarray, leaf_size: float) -> np.ndarray:
    """Keep one source point nearest each voxel center, deterministically."""
    points = _points(points)
    leaf_size = float(leaf_size)
    if leaf_size <= 0.0:
        raise ValueError('leaf_size must be positive')
    if len(points) == 0:
        return points.copy()

    keys = np.floor(points / leaf_size).astype(np.int64)
    centers = (keys.astype(np.float64) + 0.5) * leaf_size
    distances = np.sum((points - centers) ** 2, axis=1)
    order = np.lexsort((np.arange(len(points)), distances,
                        keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    first_in_voxel = np.concatenate((
        [True], np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)))
    return points[order[first_in_voxel]]


def _points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return points.reshape(0, 3)
    points = points.reshape(-1, 3)
    return points[np.all(np.isfinite(points), axis=1)]
