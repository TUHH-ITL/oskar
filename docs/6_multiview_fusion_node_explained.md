# How `multiview_fusion_node` Works

File: [`oskar_mapping/oskar_mapping/multiview_fusion_node.py`](file:///home/oskarstudent/Documents/oskar_projeckt/oskar_student/oskar_mapping/multiview_fusion_node.py)

This is the **spatial integration and clustering node** in the Phase 1 mapping pipeline — the bridge that fuses multiple, redundant observations of the same flowers across different frames into single, distinct physical landmarks.

---

## Its job in one sentence

It buffers 3D flower observations in world/map coordinates over a moving window of time and distance, groups nearby observations using DBSCAN clustering, and publishes confidence-weighted merged landmark centroids.

---

## What it needs (Inputs & Requirements)

1. **Incoming Messages:**
   * Topic: `/flowers/world_obs` (`oskar_msgs/msg/FlowerLandmarks`, from `backprojection_node`)
2. **Transform Frames (via TF2 Tree):**
   * Needs active publishers sending `/tf` messages for the transform tree (specifically: `map ➔ base_link`) to locate the robot's current position for distance-based buffer flushing.
3. **Clustering & Window Parameters:**
   * `eps` (parameter): The maximum distance between two observations to be considered in the same cluster (default: `0.08 m`).
   * `min_samples` (parameter): The minimum number of observations required to form a cluster/valid flower landmark (default: `3`).
   * `window_m` (parameter): The distance window along the robot's travel direction (X-axis) before flushing the rolling buffer (default: `4.0 m`).
   * `max_age_s` (parameter): The age window fallback in seconds to flush old observations from the buffer (default: `30.0 s`).

---

## Where output gets published (Outputs)

1. **`/flowers/landmarks`** (`oskar_msgs/msg/FlowerLandmarks`)
   * **The final merged landmark output.** Contains an array of `FlowerLandmark` messages.
   * Each `FlowerLandmark` carries:
     * `landmark_id`: Unique global landmark identifier.
     * `position`: Geometry Point ($x, y, z$) representing the confidence-weighted centroid in map coordinates.
     * `observation_count`: Total number of clustered observations.
     * `mean_confidence`: The average confidence score of all observations in the cluster.
     * `is_inferred`: Set to `False`.
     * `tree_id`: Set to `0` initially (to be assigned by `tree_assignment` downstream).

---

## Core fusion & clustering details

### 1. Two-Tiered Buffer Flushing
To prevent memory growth and avoid clustering old or distant observations, the buffer of observations is flushed via two triggers:
* **Spatial Window (Primary):** When the robot travels a distance along the X-axis exceeding `window_m` relative to `self.last_flushed_x`, the entire observation buffer is cleared.
* **Temporal Window (Fallback):** In case the robot is stationary (e.g. testing or stopped in the row), observations older than `max_age_s` seconds are removed.

### 2. DBSCAN Clustering
The node runs the **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) algorithm on all 3D coordinates currently held in the rolling buffer. This is ideal because:
* It does not require pre-specifying the number of clusters.
* It filters out isolated false-positive detections as "noise" (label `-1`).

### 3. Confidence-Weighted Centroid
For each valid cluster, rather than taking a simple geometric mean, the node calculates a **confidence-weighted centroid**. A higher-confidence detection influences the final flower position more than a low-confidence detection:
$$\text{weights}_i = \frac{\text{confidence}_i}{\sum_{j} \text{confidence}_j}$$
$$\text{Centroid} = \sum_{i} (\text{position}_i \times \text{weights}_i)$$

---

## The code, top to bottom

### 1. MultiviewFusionNode Initialization — lines 23–57
* Declares standard clustering parameters (`eps`, `min_samples`) and flushing parameters (`window_m`, `max_age_s`).
* Instantiates `tf2_ros.Buffer` and `tf2_ros.TransformListener` to track the robot's coordinates in the orchard.
* Initializes `self.observations` as a double-ended queue (`deque`) to hold the rolling window observations.
* Sets up subscription to `/flowers/world_obs` and publisher for `/flowers/landmarks` using a `BEST_EFFORT` QoS profile.

### 2. Robot Position Query — lines 59–69
* `get_robot_position_x` queries the TF tree for the latest transform from `map` to `base_link` at the message's timestamp. If the transform lookup fails due to frame mismatch or sensor lag, it safely returns `None`.

### 3. Buffer Flushing — lines 70–93
* **Lines 74–80:** Checks if the robot has traveled more than `window_m` meters since the last flush. If so, it clears the entire queue and resets the baseline `self.last_flushed_x`.
* **Lines 83–92:** Iterates through the queue, keeping only observations whose age (relative to the current timestamp) is strictly less than `max_age_s` seconds.

### 4. Observations Callback — lines 94–165
Triggered by each incoming set of observations:
* **Lines 98–99:** Fetches the robot's current coordinate and flushes the buffer.
* **Lines 102–103:** Appends the incoming flower observations along with their timestamp and robot position.
* **Lines 106–108:** Skips clustering if the buffer contains fewer observations than `min_samples`.
* **Lines 111–112:** Extracts the $(x, y, z)$ coordinates from the buffer into a NumPy array.
* **Lines 115–116:** Fits DBSCAN to the extracted positions using `eps` and `min_samples`.
* **Lines 119–125:** Groups indices of buffer observations by their cluster labels, ignoring noise (`-1`).
* **Lines 132–158:** Loops over each cluster:
  * Discards singleton clusters.
  * Computes the confidence-weighted centroid.
  * Creates a merged `FlowerLandmark` with an incremented unique ID and populates position, observation count, and mean confidence.
  * Appends the merged landmark to the list.
* **Line 161:** Publishes the resulting list on `/flowers/landmarks`.

---

## What we did (Real-World Bag Fixes)

* **Loosened `eps` 0.08 → 0.3 to get clusters to form (and why).** With the default
  `eps=0.08 m` (8 cm), clusters almost never formed: our world positions are noisy (non-metric
  depth + pose jitter), so the *same* flower scatters more than 8 cm between frames and rarely
  gets 3 observations inside the ball. The `/flowers/landmarks` stream was therefore **mostly
  empty** — a populated message only now and then, so `ros2 topic echo --once` usually caught an
  empty one (this is why fusion *looked* like it "worked then stopped" — it was timing luck, not
  a change). We raised `eps` to **0.3 m** (`-p eps:=0.3`), comfortably above the noise, so
  clusters form reliably — which we needed so the downstream `bio_sanity` actually gets input.
* **Result:** at `eps=0.3`, fusion produced 3 merged landmarks with `observation_count`
  **34 / 10 / 3** (same flower seen across that many frames and merged into one).
* **Caveat / assumption:** `eps=0.3` is deliberately *loose* and almost certainly
  **over-merges** distinct flowers (the count of 34 is the tell). It only proves the node runs —
  it is **not** a quality map. The correct `eps` is roughly the true flower-cluster size (~cm),
  which needs **metric depth (calibration)** first. `eps`/`min_samples` are clustering knobs,
  not physical calibration. (Full assumptions: `BAG_PIPELINE_BRINGUP.md` §7 / §9c.)
