# How `segmentation_node` Works

File: [`oskar_mapping/oskar_mapping/segmentation_node.py`](file:///home/oskarstudent/Documents/oskar_projeckt/oskar_student/oskar_mapping/segmentation_node.py)

This is the **flower detector node** in the Phase 1 mapping pipeline — the AI perception block that locates individual flower coordinates in the 2D image.

---

## Its job in one sentence

It reads the rectified left camera image, runs a Mask R-CNN instance segmentation model on it, and publishes the boundary pixel masks and computed 2D centroids for every detected apple flower.

---

## What it needs (Inputs & Requirements)

1. **Incoming Message:**
   * Topic: `/stereo/sync_pair`
   * Type: `oskar_msgs/msg/StereoPair` (subscribes to this but only extracts `left_image`).
2. **AI Model & Configuration:**
   * `model_path` (parameter): Absolute path to the trained Mask R-CNN model weights file (e.g., `model_final.pth`).
   * `detectron2_config.yaml`: The model architecture config file, which **must** be stored in the same parent directory as the model weights file.
3. **Detection Params:**
   * `confidence_threshold` (parameter): Detections scoring below this probability (e.g., `0.3` or `0.5`) are discarded.
   * `device` (parameter): Hardware targeting (`cuda` or `cpu`).

---

## Where output gets published (Outputs)

1. **`/flowers/masks`** (`oskar_msgs/msg/FlowerMasks`)
   * **The primary downstream output.** Contains a header and an array of individual `FlowerMask` messages.
   * Each `FlowerMask` carries:
     * `instance_id`: Unique integer within this frame.
     * `confidence`: Float model score (0.0 to 1.0).
     * `centroid_u`, `centroid_v`: Pixel coordinate of the flower center.
     * `mask`: A binary image (`mono8` encoding, 255 = flower, 0 = background) of the exact segmented flower pixels.

---

## Core segmentation details

### Centroid Computation
For each detected instance mask, the node computes the center of gravity (average pixel coordinates) to identify where the flower's center lies:
$$Centroid_u = \frac{1}{N} \sum_{i=1}^N x_i, \quad Centroid_v = \frac{1}{N} \sum_{i=1}^N y_i$$
This coordinate is passed to the depth fusion node to lock down the 3D position.

---

## The code, top to bottom

### 1. SegmentationNode Initialization — lines 23–53
* Declares standard parameters: path to model weights, model confidence threshold limit, and targeted hardware device.
* Calls `self.load_model` to compile the network configuration.
* Subscribes to `/stereo/sync_pair` with a `BEST_EFFORT` QoS profile and sets up the `/flowers/masks` publisher.

### 2. Loading the Model — lines 55–72
* Looks for the file `detectron2_config.yaml` located in the same directory as the model weights file.
* Merges the configuration file into a Detectron2 config struct (`get_cfg()`).
* Sets weights, score thresholds, and device configurations, and returns a standard `DefaultPredictor`.

### 3. Image Processing Callback — lines 74–123
When a synchronized `StereoPair` arrives:
* **Line 78:** Uses `cv_bridge` to convert the left camera frame into an OpenCV BGR image.
* **Line 81:** Feeds the image to the predictor on the GPU.
* **Lines 84–86:** Extracts prediction instances. It grabs the binary pixel masks (`pred_masks` of shape $N \times H \times W$) and confidence scores.
* **Lines 94–116:** Iterates over every detected instance:
  * Discards detections scoring below the `confidence_threshold`.
  * **Centroid Math (lines 98–103):** Runs `np.where(mask)` to find the indices of all positive flower pixels. It takes the mathematical average of the X coordinates (`centroid_u`) and Y coordinates (`centroid_v`) to find the flower's center.
  * **Formatting the mask (lines 112–114):** Converts the boolean mask array to a 0–255 `uint8` array and packages it as a single-channel `mono8` ROS Image message.
  * Appends the completed `FlowerMask` to the collective message array and publishes it on `/flowers/masks`.

---

## What we did (Real-World Bag Fixes)

* **Detectron2 CPU Build Workaround:** The system CUDA toolkit version was mismatched, causing source compilation of `detectron2` to crash. We compiled `detectron2` in CPU-only mode:
  ```bash
  CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0 pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
  ```
  Even though the wrapper package is compiled for CPU, it still performs high-speed GPU forward passes through PyTorch's native CUDA backend, achieving ~1.26 seconds inference times.
* **Synchronized Feeder Rate Integration:** Since the Mask R-CNN segmentation model runs slower than standard image publication rates, we lowered the feeder rate to **0.3 Hz** in tests, ensuring the disparity node and segmentation node finish processing the same frame to satisfy depth synchronization.
* **Result:** detected **~37 flowers** on a full-res image standalone (1.26 s on GPU), and
  **~42 flowers per frame** in-pipeline on `/flowers/masks`. (We run on the 0.25-downscaled
  image, so small/distant flowers are missed vs full res — fewer detections than full-res.)
* **Weights used:** `…/synthetic_apple_flowers/results/exp_bhangale/train/model_final.pth`
  (GeneralizedRCNN Mask R-CNN, 1 class = flower), with its `detectron2_config.yaml` beside it.
* **GPU split (2026-06-18):** run with `CUDA_VISIBLE_DEVICES=1` so segmentation uses GPU 1 and
  doesn't contend with disparity (GPU 0) — both on one GPU roughly halves throughput.
* **confidence 0.3, not 0.08:** the `/flowers/masks` message embeds a **full-res mask per
  flower** (~1.5 MB each). At 0.08 it emits ~99 junk detections → ~150 MB message that's nearly
  undeliverable; 0.3 gives ~30–40 → ~64 MB.
* **QoS fix (2026-06-18):** both the `/stereo/sync_pair` subscriber and the `/flowers/masks`
  publisher were `BEST_EFFORT`, which **drops these large messages** on localhost. Changed both
  to **`RELIABLE`** (masks publisher uses `depth=1` to avoid hoarding the huge messages). The
  proper long-term fix is to not ship a full-frame mask per flower (use bbox/RLE). See
  `BAG_PIPELINE_BRINGUP.md` §11b.
