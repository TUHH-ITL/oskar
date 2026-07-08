# How `depth_fusion_node` Works

File: [`oskar_mapping/oskar_mapping/depth_fusion_node.py`](file:///home/oskarstudent/Documents/oskar_projeckt/oskar_student/oskar_mapping/depth_fusion_node.py)

This is the **3D localized fusion node** in the Phase 1 mapping pipeline — the bridge connecting the 2D flower boundaries with the 3D depth map.

---

## Its job in one sentence

It time-synchronizes the flower mask coordinates and the computed depth map, extracts the median depth values belonging strictly to each flower mask, and projects the 2D pixel centroids into 3D space relative to the camera lens.

---

## What it needs (Inputs & Requirements)

1. **Incoming Messages (Synchronized):**
   * Topic 1: `/stereo/depth` (`sensor_msgs/msg/Image`, from `disparity_node`)
   * Topic 2: `/flowers/masks` (`oskar_msgs/msg/FlowerMasks`, from `segmentation_node`)
   * Both are matched together using an `ApproximateTimeSynchronizer` with a slop of **0.05 seconds**.
2. **Camera Calibration (Independent):**
   * Topic: `/static_camera_info_1` (configured via `left_info_topic`) containing the camera projection matrix.
3. **Filtering & Erosion Params:**
   * `erosion_radius_px` (parameter): Number of pixels to erode the binary flower mask boundaries (default: `3 px`).
   * `depth_std_threshold` (parameter): Threshold for filtering out depth calculations with high standard deviation (default: `0.15`).
   * `min_valid_pixels` (parameter): Minimum number of un-eroded pixels containing valid depth readings needed to keep the detection (default: `5`).

---

## Where output gets published (Outputs)

1. **`/flowers/detections_3d`** (`oskar_msgs/msg/FlowerDetections3D`)
   * **The primary downstream output.** Contains an array of `FlowerDetection3D` objects.
   * Each object represents a flower in **3D Camera Coordinates** ($x, y, z$ in meters relative to the left camera lens center) along with flags:
     * `position`: Geometry Point (`x` is lateral offset, `y` is vertical offset, `z` is distance forward).
     * `low_confidence`: Boolean set to `True` if depth readings are noisy.
     * `depth_std_ratio`: The noise ratio ($\sigma / \text{median}$) computed for that flower.

---

## Core fusion & math details

### 1. Mask Boundary Erosion
Flower borders often overlap background branches or sky, causing huge errors if we average depth over the raw mask. The node uses OpenCV to erode the mask edges:
```python
mask_eroded = cv2.morphologyEx(mask_binary, cv2.MORPH_ERODE, kernel)
```
Depth is only sampled from this safe inner region.

### 2. Median Depth Filtering
The node extracts all depth values inside the eroded mask, filters out `NaN` values, and finds the **median** rather than the mean. The median is highly robust to remaining outliers.

### 3. Outlier Flagging
The node evaluates depth standard deviation ($\sigma$) divided by the median depth. If the ratio exceeds `depth_std_threshold`, the reading is flagged as `low_confidence = True`:
$$\text{depth\_std\_ratio} = \frac{\sigma}{\text{Median Depth}}$$

### 4. 3D Pin-hole Camera Backprojection
The pixel centroid $(u, v)$ is projected to camera-relative 3D coordinates $(x_{cam}, y_{cam}, z_{cam})$ using focal lengths ($f_x, f_y$) and center offsets ($c_x, c_y$):
$$z_{cam} = \text{Median Depth}$$
$$x_{cam} = \frac{u - c_x}{f_x} \times z_{cam}$$
$$y_{cam} = \frac{v - c_y}{f_y} \times z_{cam}$$

---

## The code, top to bottom

### 1. DepthFusionNode Initialization — lines 23–65
* Declares standard parameters: camera calibration info topic, boundary erosion radius, depth noise standard deviation ratio threshold, and the minimum size in pixels needed to process depth.
* Sets up a message filter Subscriber for `/stereo/depth` and `/flowers/masks`, combining them via `ApproximateTimeSynchronizer` with a slop of **0.05 seconds** and registering `self.fusion_callback`.
* Subscribes independently to `left_info_topic` (since calibration properties are static, they do not need time-synchronization).

### 2. Sourcing Calibration Parameters — lines 67–70
* `info_callback` caches the incoming `/static_camera_info_1` message into `self.left_info`.

### 3. The Fusion Callback — lines 71–149
When synchronized depth and masks arrive:
* **Lines 74–76:** Discards the frame if no camera calibration parameters are cached yet.
* **Line 79:** Converts the depth map message to a NumPy float32 array.
* **Lines 82–86:** Reconstructs focal lengths (`fx, fy`) and principal point centers (`cx, cy`) from the cached camera matrix.
* **Lines 93–96:** Generates an elliptical morph kernel of size $(2 \cdot \text{erosion\_radius\_px} + 1)$ using OpenCV.
* **Lines 99–139:** Loop through every flower mask inside `masks_msg.flowers`:
  * **Erosion (lines 102–106):** Converts the mask to a binary image and applies `cv2.MORPH_ERODE` to strip off boundary pixels (removing edges that might contain background depth).
  * **Sampling (lines 109–110):** Extracts depth values matching the positive coordinates of the eroded mask, discarding any `NaN` values.
  * **Size Check (lines 112–113):** Skips the flower entirely if there are fewer than `min_valid_pixels` left after erosion.
  * **Stats (lines 116–120):** Calculates the median depth (`z_cam`), standard deviation (`depth_std`), and the ratio. If the ratio exceeds `depth_std_threshold`, it flags `low_confidence = True`.
  * **Projection Math (lines 122–127):** Applies the pin-hole camera equations to convert the 2D image coordinate centroid (`centroid_u, centroid_v`) to camera coordinates ($x, y, z$ in meters).
  * Appends the finalized `FlowerDetection3D` and publishes the list on `/flowers/detections_3d`.

---

## What we did (Real-World Bag Fixes)

* **Info Topic Path Correction:** The default topic parameter `left_info_topic` was configured for simulation as `/rgb_camera_1/camera_info`. We updated the parameter configuration to look at `/static_camera_info_1` to read calibration values.
* **Throughput Synchronization:** Because disparity calculation takes ~2 seconds and segmentation takes ~1 second, they are published at different rates. If the image feeder publishes at normal camera speeds (e.g. 5 Hz), the synchronizer will drop most frames because matching timestamps cannot be paired. We set the feeder rate to **0.3 Hz**, giving the GPU enough time to process and output matching stamps for both topics.
* **Result:** `/flowers/detections_3d` carries flowers in 3D (camera frame); the
  `low_confidence` / `depth_std_ratio` filter correctly flags clean (~0.005–0.02) vs noisy
  detections. One garbage outlier was seen (z≈24000 m from a near-zero-disparity pixel) — left
  for the downstream multiview/bio stages to reject.
* **Assumption:** positions are **non-metric** (inherited from disparity's synthetic calibration);
  correct scale needs real calibration. (Full assumptions: `BAG_PIPELINE_BRINGUP.md` §7.)
* **QoS fix (2026-06-18):** the depth + masks subscribers were `BEST_EFFORT`, which **drops the
  large depth (~6 MB) and masks (~64 MB) messages** → the `ApproximateTimeSynchronizer` never
  paired a depth with a mask, so **0 detections**. Changed the subscribers to **`RELIABLE`**;
  `/flowers/detections_3d` then flowed at the full **0.3 Hz** with real 3D flowers. This was the
  final blocker for an end-to-end run. See `BAG_PIPELINE_BRINGUP.md` §11b.
