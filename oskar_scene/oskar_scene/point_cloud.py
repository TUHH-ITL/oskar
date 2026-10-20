"""ROS PointCloud2 conversion helpers."""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField


def cloud_to_points(cloud: PointCloud2) -> np.ndarray:
    """Decode finite XYZ values from a PointCloud2 message."""
    from sensor_msgs_py import point_cloud2

    values = np.asarray(point_cloud2.read_points(
        cloud, field_names=('x', 'y', 'z'), skip_nans=False))
    if values.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if values.dtype.names:
        points = np.column_stack([
            values['x'], values['y'], values['z']]).astype(np.float64)
    else:
        points = values.astype(np.float64).reshape(-1, 3)
    return points[np.all(np.isfinite(points), axis=1)]


def points_to_cloud(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """Encode XYZ points into a dense PointCloud2 with the supplied header."""
    from std_msgs.msg import Header
    from sensor_msgs_py import point_cloud2

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    header = Header(frame_id=frame_id, stamp=stamp)
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    return point_cloud2.create_cloud(header, fields, points.tolist())
