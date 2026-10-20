from sensor_msgs.msg import PointCloud2


def test_snapshot_contract_preserves_header():
    cloud = PointCloud2()
    cloud.header.frame_id = 'camera_optical_frame'
    cloud.header.stamp.sec = 42
    cloud.header.stamp.nanosec = 7
    assert cloud.header.frame_id == 'camera_optical_frame'
    assert cloud.header.stamp.sec == 42
    assert cloud.header.stamp.nanosec == 7
