# How `backprojection_node` Works

File: [`oskar_mapping/oskar_mapping/backprojection_node.py`](file:///home/oskarstudent/Documents/oskar_projeckt/oskar_student/oskar_mapping/backprojection_node.py)

This is the **coordinate transformation node** in the Phase 1 mapping pipeline — the spatial compiler that maps localized camera detections into absolute coordinates in the orchard.

---

## Its job in one sentence

It listens to 3D flower positions in camera-relative coordinates, queries the ROS 2 TF2 library to locate the exact position of the camera in the orchard at the time the photo was taken, and transforms the flower coordinates into fixed world/map coordinates.

---

## What it needs (Inputs & Requirements)

1. **Incoming Message:**
   * Topic: `/flowers/detections_3d` (`oskar_msgs/msg/FlowerDetections3D`, from `depth_fusion_node`)
2. **Transform Frames (via TF2 Tree):**
   * `camera_frame` (parameter): Target frame representing the camera lens (defaults to `camera_left_optical`).
   * `map_frame` (parameter): Static coordinate origin for the entire orchard (defaults to `map`).
   * Needs active publishers sending `/tf` messages for the transform tree (namely: `map ➔ base_link` and `base_link ➔ camera_left_optical`).

---

## Where output gets published (Outputs)

1. **`/flowers/world_obs`** (`oskar_msgs/msg/FlowerLandmarks`)
   * **The primary downstream output.** Contains an array of `FlowerLandmark` messages representing flowers mapped to static world coordinates.
   * Each `FlowerLandmark` carries:
     * `landmark_id`: Unique global landmark identifier.
     * `position`: Geometry Point ($x, y, z$) in fixed map coordinates.
     * `observation_count`: Set to `1` initially (incremented by downstream nodes).
     * `mean_confidence`: Confidence propagated from the detector.
     * `is_inferred`: Set to `False` (inferred flowers are generated later by `bio_sanity`).

---

## Core math & transform details

### Time-Stamped TF Lookup
As the robot moves, camera transforms change constantly. To avoid spatial shearing, the node looks up the transform at the exact nanosecond the camera shutter was open (`detections_msg.header.stamp`):
```python
transform = self.tf_buffer.lookup_transform(
    self.map_frame, 
    self.camera_frame,
    detections_msg.header.stamp,
    timeout=rclpy.duration.Duration(seconds=0.1)
)
```
This returns the rotation matrix $R_{cam}^{map}$ and translation vector $T_{cam}^{map}$.

### 3D Coordinate Translation
The point in the camera frame $P_{cam}$ is converted to the map frame $P_{map}$ via:
$$P_{map} = R_{cam}^{map} \times P_{cam} + T_{cam}^{map}$$
This is calculated inside the node using the utility function:
```python
point_map = do_transform_point(point_cam, transform)
```

---

## The code, top to bottom

### 1. BackprojectionNode Initialization — lines 21–47
* Declares standard parameters: `camera_frame` (origin of 3D points) and `map_frame` (target frame coordinates).
* Instantiates `tf2_ros.Buffer` and its associated `tf2_ros.TransformListener`, which automatically binds to the node executor and listens to `/tf` and `/tf_static` in the background to build the coordinate frame tree.
* Sets up a subscriber to `/flowers/detections_3d` and a publisher on `/flowers/world_obs`.

### 2. Detections Callback — lines 49–101
Runs for every incoming message array:
* **Lines 58–61:** Creates an empty output array `FlowerLandmarks` with the header set to match the static map frame coordinate system.
* **Lines 63–70:** Queries the TF listener buffer for the coordinate transformation from `map` to `camera_left_optical` at the exact message stamp. It sets a small timeout (0.1 seconds) to wait if there is minor sensor lag before throwing a `TransformException`.
* **Lines 70–94:** Iterates over every 3D coordinate detection in the message:
  * **Coordinate Packaging (lines 72–75):** Wraps the detection coordinates inside a ROS `PointStamped` container, stamping it with the camera frame ID.
  * **Math Transformation (lines 78–82):** Calls `do_transform_point` to execute the vector multiplication ($P_{map} = R \cdot P_{cam} + T$), projecting the point into map coordinates.
  * **Landmark creation (lines 85–94):** Instantiates a `FlowerLandmark` message, increments the unique landmark tracker ID, copies the transformed geometry point, and appends it to the output list.
* **Line 97:** Publishes the finished landmarks onto `/flowers/world_obs`.

---

## What we did (Real-World Bag Fixes)

* **GNSS Trajectory-to-TF Node (`gnss_pose_tf_node.py`):** The bag file lacks a fused TF tree (`map ➔ base_link`). To solve this:
  1. We ran `extract_pose_trajectory.py` to parse RTK GNSS `/ublox/fix` and convert coordinates to local ENU (East-North-Up) values relative to the tree map centroid.
  2. Because the IMU yaw suffered a massive 113° drift step during U-turns, we designed the pose node to calculate heading solely by smoothing the GNSS course-over-ground trajectory.
  3. The custom `gnss_pose_tf_node.py` interpolates these smoothed coordinates at each camera timestamp and publishes the active `/tf` tree from `map ➔ base_link` and a static transform from `base_link ➔ camera`.
* **Calibration Gap Diagnostic:** Detections are currently placed **5 to 8 meters away** from the actual tree locations. While the pipeline operates correctly, this error is caused by using a guessed, synthetic `CameraInfo` and mounting transformation. Real stereo intrinsics/extrinsics and camera mount measurements are required from the supervisor to align the maps metrically.
