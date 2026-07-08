# How `disparity_node` Works

File: [`oskar_mapping/oskar_mapping/disparity_node.py`](file:///home/oskarstudent/Documents/oskar_projeckt/oskar_student/oskar_mapping/disparity_node.py)

This is the **second node** in the Phase 1 mapping pipeline — the depth estimation engine that translates flat stereo camera views into 3D space.

---

## Its job in one sentence

It takes the synchronized left and right images, uses a deep learning stereo model to calculate the horizontal pixel shift (disparity) for each pixel, and converts those shifts into absolute depth (meters) using camera geometry.

---

## What it needs (Inputs & Requirements)

1. **Incoming Message:**
   * Topic: `/stereo/sync_pair`
   * Type: `oskar_msgs/msg/StereoPair` (published by `stereo_sync_node`) containing synchronized left and right images plus calibration info.
2. **AI Model & Directory:**
   * `ffm_dir` (parameter): Path to the **Fast-FoundationStereo** repository folder.
   * `model_dir` (parameter): Path to the PyTorch weights checkpoint (e.g., `model_best_bp2_serialize.pth`).
3. **Geometry Params:**
   * `baseline_m` (parameter): Physical separation distance in meters between the two cameras (physical value is **0.13 m** / 13 cm).
   * `scale` (parameter): Scaling factor to shrink image resolution for GPU inference speed. For the current full-resolution bag run, keep this at `1.0`.
   * `tile_height` / `tile_overlap` / `tile_width` / `tile_x_overlap` (parameters): 2D rectangular tiling parameters for full-resolution images. This runs the model on overlapping rectangular patches and stitches them back into one disparity image. It uses a horizontal overlap (`tile_x_overlap`) of at least `max_disp` to preserve the stereo search range across columns.

---

## Where output gets published (Outputs)

1. **`/stereo/disparity`** (`stereo_msgs/msg/DisparityImage`)
   * Used for visualization in RViz. Stores the computed pixel shift map.
2. **`/stereo/depth`** (`sensor_msgs/msg/Image` encoded as `32FC1`)
   * **The primary downstream output.** A matrix of float values where each pixel represents distance in meters.
3. **`/stereo/points`** (`sensor_msgs/msg/PointCloud2`)
   * A dense 3D point cloud of the visible orchard scene, projected using focal length and depth. Highly useful for debugging scene structure in RViz.

---

## Core math & pipeline details

### 1. Depth from Disparity
For each pixel, absolute depth $z$ is computed using the standard camera model:
$$z = \frac{f_x \times b}{d}$$
* $f_x$: Focal length in pixels (read from camera calibration).
* $b$: Camera baseline distance in meters (from `baseline_m`).
* $d$: Disparity value in pixels (output by the stereo model).

### 2. Startup Warmup
On launch, the node executes `warmup_model` by feeding dummy images through PyTorch. This triggers CUDA compilation/memory allocation before any ROS topics arrive, preventing massive frame drops on the first live messages.

### 3. Full-resolution tiling
Full 5328x4608 stereo frames are too large for Fast-FoundationStereo to process as one image on the available GPU memory. In addition, the full width of 5328 can cause cuDNN errors (`CUDNN_STATUS_NOT_SUPPORTED`) even when contiguous. To solve this, the node supports **2D rectangular tiling** using `tile_height`, `tile_overlap`, `tile_width`, and `tile_x_overlap`.

Stereo disparity matching is asymmetric: matching pixels in the right image are always shifted to the left by up to `max_disp` relative to the left image. To keep the matching range intact across vertical tile columns, the horizontal tiling uses an asymmetric left overlap (`tile_x_overlap`) which defaults to `max_disp` (e.g., `384`). For every pixel in the core stitched region, its potential match is guaranteed to reside within the right tile. 

The vertical tiling uses standard symmetric `tile_overlap` (e.g., `96`). Before inference, each tile is forced into contiguous memory using `np.ascontiguousarray` to satisfy cuDNN.

### 4. Performance & Processing Rate
When running at `scale:=1.0`, splitting a 5328x4608 image into tiles of size 768x1024 with 96px vertical overlap and 384px horizontal overlap results in exactly **72 tiles** (8 vertical $\times$ 9 horizontal steps). 

Processing 72 tiles sequentially on a single GPU takes approximately 60–90 seconds per frame, resulting in an average topic rate of **~0.013 – 0.017 Hz**. While too slow for real-time applications, this is the expected and correct execution profile for achieving mathematically sound full-resolution depth mapping on consumer hardware.



---

## The code, top to bottom

### 1. FFM Shadow Patching — lines 8–13
The ROS workspace folder has a local package named `statistics` that shadows the default Python `statistics` library. PyTorch requires the native standard library to work, so lines 11–13 search `sys.path` and strip out the shadow package before importing `torch`.

### 2. FastFoundationStereoModel Class — lines 30–101
This is the PyTorch model wrapper:
* **`__init__` (lines 33–56):** Appends `ffm_dir` to the system environment path, loads the serialized weights on the target GPU/CPU using `torch.load()`, sets iterations/disp configurations, and sets the model to evaluation (`eval()`) mode.
* **`forward` (lines 58–101):** Converts OpenCV images (BGR format) to RGB, pads the boundaries to a multiple of 32 (needed for the model structure), runs inference inside a half-precision autocast (`torch.amp.autocast`), unpads the output, and returns it as a NumPy float32 array.

### 3. DisparityNode Class Initialization — lines 104–171
* Declares standard parameters including directory paths, inference scale factor, and physical camera baseline.
* Instantiates `FastFoundationStereoModel` and calls `self.warmup_model()`.
* Sets up a subscriber to `/stereo/sync_pair` and initializes publishers for the disparity map, depth image, and 3D point cloud.

### 4. Warm-up Sequence — lines 173–187
Generates a dummy NumPy random array of the configured resolution and pushes it through the model. Since PyTorch CUDA compilation takes 15–30 seconds on the first run, this ensures compile latency happens at node startup, not when live camera frames are received.

### 5. Stereo Processing Callback — lines 188–283
When a new synchronized image pair is received:
* **Lines 199–213:** Scales the images down using OpenCV `resize` if `scale < 1.0` to speed up calculation.
* **Line 214:** Obtains the raw disparity matrix from the model. If `tile_height > 0`, this happens band-by-band and is stitched into one full-frame disparity matrix.
* **Lines 217–222:** Scales the disparity matrix back up to full camera resolution using `INTER_NEAREST` to preserve boundaries.
* **Lines 224–242:** Extracts camera intrinsics focal length (`fx`) and computes the baseline. If no baseline is present in the `CameraInfo` matrix, it falls back to the parameter override value (`baseline_m`). It then divides `(fx * baseline)` by the disparity to output depth in meters.
* **Lines 244–280:** Publishes the `DisparityImage` (scaled 32-bit floats), the raw depth map, and calls `_depth_to_pointcloud` to project all valid depth pixels into a dense 3D PointCloud (`PointCloud2`) for RViz visualization.

---

## What we did (Real-World Bag Fixes)

* **Left/Right Swap Correction:** Sourced images from the top camera (`SAMSON3`) as Left (`/rgb_static_1`) and the bottom camera (`SAMSON4`) as Right (`/rgb_static_2`). Sourcing them backwards originally caused inverted disparity computation and scrambled depth clouds.
* **Full-resolution tiling:** The current run keeps feeder/disparity scale at `1.0` and uses 2D rectangular tiling with `tile_height:=768`, `tile_overlap:=96`, `tile_width:=1024`, and `tile_x_overlap:=384`. This avoids global downscaling while keeping each GPU inference chunk small enough to fit.
* **Older resolution scaling test:** Earlier testing used `scale:=0.25`, which required `max_disp:=192`. Full scale needs more range, but this FFM checkpoint fails at `max_disp:=512`; use `384` unless a larger-disparity checkpoint is available.
* **Environment Isolation:** Sourced `~/oskar_venv` to isolate the model's PyTorch/CUDA libraries and force NumPy < 2.0, resolving conflicts with ROS `cv_bridge`.
* **Result:** `/stereo/depth` publishes a stitched full-resolution depth map; the `/stereo/points`
  cloud shows coherent orchard 3D structure after the left/right swap.
* **Calibration:** the feeder now defaults to the real SAMSON3/SAMSON4 stereo calibration files.
  `baseline_m` remains as a fallback override if a CameraInfo message does not encode baseline.
  Camera-to-platform mounting calibration is still a separate blocker for world-frame accuracy.
* **QoS fix (2026-06-18):** the `/stereo/sync_pair` subscriber was `BEST_EFFORT`, which silently
  **drops the ~9 MB stereo message** on localhost → the node was starved (GPU idle, ~1 frame /
  45 s). Changed the subscriber to **`RELIABLE`**; it now keeps up at the feeder rate. See
  `BAG_PIPELINE_BRINGUP.md` §11b.
