# oskar_sensing

This package owns the camera boundary. `snapshot_node` consumes a source
`sensor_msgs/msg/PointCloud2` topic and republishes exactly one fresh cloud from
`capture_snapshot` to `/scene/input_cloud`. It preserves the cloud frame and
stamp; `oskar_scene` does not import ZED or CUDA APIs.

`zed_x_live.launch.py` and `zed_x_svo.launch.py` wrap `zed_wrapper` and feed
the same snapshot node. The registered cloud topic and wrapper argument names
remain launch-configurable because they must be checked against the deployed
Orin image.
