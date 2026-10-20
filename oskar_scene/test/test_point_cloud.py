import numpy as np
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from oskar_scene.point_cloud import cloud_to_points


def test_cloud_to_points_decodes_jazzy_structured_array():
    cloud = point_cloud2.create_cloud_xyz32(
        Header(frame_id='fixture_sensor'),
        [(1.0, 2.0, 3.0), (np.nan, 4.0, 5.0)])

    points = cloud_to_points(cloud)

    np.testing.assert_allclose(points, [[1.0, 2.0, 3.0]])