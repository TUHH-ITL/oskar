# How `stereo_sync_node` Works

File: [`oskar_mapping/oskar_mapping/stereo_sync_node.py`](../oskar_mapping/oskar_mapping/stereo_sync_node.py)

This is the **first node** in the Phase 1 mapping pipeline — the funnel that turns
two messy camera streams into clean, paired input for depth estimation.

---

## Its job in one sentence

Two cameras (left + right) produce a constant stream of images. This node's job is to
**grab one left image and one right image that were taken at the same instant, clean
them up (rectify), and bundle them into a single message** so the next node (disparity)
can compare them to compute depth.

It's the first box in the pipeline diagram — the funnel between the raw cameras and the
rest of the system.

**Input:** `/rgb_static_1`, `/rgb_static_2` (+ their `camera_info` topics)
**Output:** `/stereo/sync_pair` — time-matched, rectified left/right images with
calibration attached.

---

## The four concepts it deals with

### 1. Why pairing is hard (time synchronization)
The left and right cameras each publish on their own topic, independently, several times
a second. Their messages never arrive at the exact same millisecond. If you naively
grabbed "the latest left" and "the latest right," they might be from slightly different
moments — and for stereo depth, even a tiny time mismatch while the robot is moving ruins
the 3D math. So the node uses a **time synchronizer** that reads the timestamp baked into
each image and only fires when it finds a left+right pair taken close enough together.

### 2. Rectification
Real camera lenses distort images (straight lines bow), and the two cameras are never
*perfectly* aligned. **Rectification** is the geometric correction that undistorts both
images and lines them up so that a point in the left image sits on the exact same
horizontal row in the right image. The disparity node depends on this. The recipe for the
correction comes from the camera's calibration data.

### 3. `camera_info` (the calibration)
Alongside the image topics, each camera publishes a `camera_info` topic carrying its
calibration: the lens matrix, distortion coefficients, etc. This data is **essentially
constant** — it doesn't change frame to frame, so the node only needs to read it **once**.

### 4. QoS (Quality of Service)
In ROS 2, a subscriber and publisher must agree on delivery rules ("QoS") or they won't
connect. The two relevant settings: `RELIABLE` (re-send until delivered, like TCP) vs
`BEST_EFFORT` (fire and forget, like UDP). **Mismatched QoS is one of the most common
"why is my node receiving nothing?" bugs in ROS** — keep it in mind.

---

## The code, top to bottom

### Setup — lines 26–33
Declares 8 **parameters** (topic names, camera frame names, and `sync_slop`). Parameters
are knobs you can change from the config file without editing code — which is why
`config/mapping_params_sim.yaml` can point the node at different topic names.
`sync_slop: 0.05` means "two images count as a pair if their timestamps are within 0.05
seconds of each other."

### The clever design — lines 49–73
It handles `camera_info` and `images` **differently**:
- **Camera info** (lines 55–60): plain subscriptions that fire once and then get ignored.
  Because calibration is static, there's no reason to sync it.
- **Images** (lines 68–73): wrapped in an `ApproximateTimeSynchronizer`. This is the
  time-matching machine — it watches both image topics and calls `_img_callback` only
  when it has a matched left+right pair.

The docstring at the top (lines 2–8) explains *why*: syncing only the 2 image topics is
far more robust than trying to sync all 4 topics, because in Isaac Sim (and often in real
bags) the `camera_info` timestamps don't line up cleanly with the image timestamps.

### Receiving calibration — lines 87–97
When the first left (or right) `camera_info` arrives, it's cached and never updated again
(the `if self._left_info is None` guard). Each time one arrives, it tries to build the
rectification recipe.

### Building the rectification maps — lines 99–133
Once *both* calibrations are in, it pulls out the camera matrices and distortion values.
Then an optimization at lines 116–123: it checks whether the images are **already
rectified** (zero distortion, identity rotation — which simulators often provide for
free). If so, it skips the correction entirely. Otherwise (lines 125–130) it precomputes
"remap tables" with OpenCV — doing this once now so every future frame can be corrected
fast. Either way it sets `self._maps_ready = True`, the green light to start processing
images.

### The per-pair workhorse — lines 139–176
Runs for every synchronized left+right pair:
- **Lines 140–145:** if calibration hasn't arrived yet, it **drops the frame** and warns
  (throttled to once every 2 seconds so it doesn't spam). This is the single most
  important gotcha — see below.
- **Lines 148–159:** converts the ROS image into an OpenCV image (`cv_bridge` does this),
  then applies the rectification remap if needed.
- **Lines 161–173:** packs the cleaned left image, cleaned right image, *and* both
  calibrations into one `StereoPair` message and publishes it on `/stereo/sync_pair`.
  Bundling the calibration with the images means the downstream disparity node gets
  everything it needs in one package.

### The bottom — lines 179–187
Standard ROS 2 boilerplate. `rclpy.init()` starts ROS, `rclpy.spin(node)` keeps the node
alive and listening forever (this is the "it just sits there" behavior you'll see when
you run it). **Every node in the pipeline ends with this same pattern.**

---

## ⚠️ The one gotcha to remember

**The node produces nothing until BOTH `camera_info` messages have arrived.**

If you ever see it stuck on `"camera info not yet available — dropping frame"`, that's the
cause. When testing with a bag file, the bag **must** contain the `camera_info` topics (or
you must publish them separately from a calibration file), or `/stereo/sync_pair` will
stay empty and the whole pipeline downstream will be silent.

---

## What we did (Real-World Bag Fixes)

* **Left/right swap.** The bag's cameras are named TOP (`SAMSON3`) / BOTTOM (`SAMSON4`); we feed
  TOP as left (`/rgb_static_1`) and BOTTOM as right (`/rgb_static_2`) via
  `-p left_image_topic:=/rgb_static_1 -p right_image_topic:=/rgb_static_2`. Feeding them the
  other way scrambled disparity downstream (see the disparity doc).
* **Real `camera_info` by default.** The feeder now loads the SAMSON3/SAMSON4 stereo calibration
  YAMLs and publishes real K/D/R/P on `/static_camera_info_1/2`.
* **Result:** `/stereo/sync_pair` publishes at ~feeder rate with left+right sharing an identical
  timestamp (sync correct), both images attached, and both camera_infos attached. With real
  calibration, this node builds rectification maps instead of skipping remap.
* **Remaining assumption:** camera-to-platform mounting is still separate from stereo
  rectification and must be confirmed before trusting world-frame flower locations.
