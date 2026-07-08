# Running the OSKAR Offline Mapping Pipeline

<style>
.copy-code-button {
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  padding: 0.25rem 0.5rem;
  border: 1px solid #d0d7de;
  border-radius: 6px;
  background: #f6f8fa;
  color: #24292f;
  cursor: pointer;
  font-size: 0.75rem;
}
</style>
<script>
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("pre").forEach((pre) => {
    pre.style.position = "relative";
    pre.style.paddingTop = "2.25rem";

    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-code-button";
    button.textContent = "Copy";
    button.addEventListener("click", async () => {
      const code = pre.querySelector("code")?.innerText || pre.innerText;
      await navigator.clipboard.writeText(code.trimEnd());
      button.textContent = "Copied!";
      setTimeout(() => {
        button.textContent = "Copy";
      }, 1500);
    });
    pre.appendChild(button);
  });
});
</script>

Runbook for running the modular offline apple flower mapping pipeline.

The offline pipeline runs outside ROS 2 using the pre-extracted trajectory (`pose_trajectory.npz`) and image folders, eliminating serialization latency and message drops.

> **Always source the environment in your terminal before running scripts:**
>
> ```bash
> source ~/ros2_ws/src/oskar/oskar_env.sh
> ```

---

## 1. Production Mode: End-to-End Run (In-Memory)

In production mode, the entire pipeline runs in memory for maximum speed and efficiency. It does **not** write intermediate files to disk, and outputs only the final `thinning_plan.json` under its specific subfolder.

By default, the pipeline runs on the first **10** frames. You can specify a different number of frames to process via the `--max-frames` argument (pass `-1` to run on the entire sequence of 354 images):

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
cd ~/ros2_ws/src/oskar/oskar_mapping/offline_pipeline

# Run on default 10 frames:
python3 run_pipeline.py

# Run on a custom number of frames (e.g. 50):
python3 run_pipeline.py --max-frames 50

# Run on the entire dataset:
python3 run_pipeline.py --max-frames -1
```

*Expected Output Directory:*

* [`offline_pipeline/data/step8_thinning_decision/thinning_plan.json`](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/data/step8_thinning_decision/thinning_plan.json)

---

## 2. Debugging Mode: Step-by-Step Run

In debugging mode, you can run each node's logic individually to inspect intermediate files. Each script automatically **clears its dedicated folder** of any old files before saving new outputs. Each script also loads its inputs from the output of the preceding step.

> **Note**: For Steps 1 to 4, you can append `--max-frames <N>` to process the first N frames, or use `--start-frame <X>` and `--end-frame <Y>` to process a specific range of frames (inclusive, e.g. `--start-frame 100 --end-frame 150`). Subsequent steps (Steps 5 to 8) will automatically process all generated files found in the previous step's folder.

### Step 1: Instance Segmentation

You have two script options depending on whether you need speed or maximum resolution recall:

#### Option A: Standard Fast Version (Default)
Runs a single-pass inference on the full image. Best for processing the complete 354 image sequence quickly:
```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
cd ~/ros2_ws/src/oskar/oskar_mapping/offline_pipeline

# Run standard segmentation:
python3 step1_segmentation.py --start-frame 100 --end-frame 110
```

#### Option B: Tiled Version
Runs overlapping 640x640 tiled inference with NMS. Best for high-recall testing on small batches:
```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
cd ~/ros2_ws/src/oskar/oskar_mapping/offline_pipeline

# Run tiled segmentation:
python3 step1_segmentation_tiled.py --start-frame 100 --end-frame 110
```

> [!TIP]
> **Saving Visuals Instantly without NPZ Compression**:
> Running tiled mode with low thresholds outputs hundreds of masks, which causes `np.savez_compressed` to take 2-3 minutes per frame. 
> To test the tiled script visually in under **10 seconds**, open [config.py](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/config.py) and temporarily set:
> * `SEG_SAVE_NPZ = False`
> * `SEG_SAVE_VISUALIZATIONS = True`

*Outputs masks to:* [`offline_pipeline/data/step1_segmentation/`](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/data/step1_segmentation/)

### Step 2: Disparity Estimation

Runs batched FFM disparity on the cropped flower patches.

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
cd ~/ros2_ws/src/oskar/oskar_mapping/offline_pipeline
python3 step2_disparity.py --max-frames -1
```

*Outputs depth maps to:* [`offline_pipeline/data/step2_disparity/`](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/data/step2_disparity/)

### Step 3: Depth and Mask Fusion

Projects the flower centroids into 3D coordinates in the left camera frame.

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
cd ~/ros2_ws/src/oskar/oskar_mapping/offline_pipeline
python3 step3_depth_fusion.py --max-frames -1
```

*Outputs 3D camera coordinates to:* [`offline_pipeline/data/step3_depth_fusion/`](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/data/step3_depth_fusion/)

### Step 4: Backprojection to World

Applies interpolated trajectory poses to transform 3D camera points into ENU world coordinates.

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
cd ~/ros2_ws/src/oskar/oskar_mapping/offline_pipeline
python3 step4_backprojection.py --max-frames -1
```

*Outputs world observations to:* [`offline_pipeline/data/step4_backprojection/world_obs.npy`](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/data/step4_backprojection/)

### Step 5: Multiview Fusion

Runs DBSCAN clustering on world coordinates to merge duplicates across frames.

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
cd ~/ros2_ws/src/oskar/oskar_mapping/offline_pipeline
python3 step5_multiview_fusion.py --max-frames -1
```

*Outputs merged landmarks to:* [`offline_pipeline/data/step5_multiview_fusion/landmarks.npy`](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/data/step5_multiview_fusion/)

### Step 6: Biology Priors (Sanity check)

Infers missing flowers for partially occluded flower corymbs.

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
cd ~/ros2_ws/src/oskar/oskar_mapping/offline_pipeline
python3 step6_bio_sanity.py --max-frames -1
```

*Outputs corrected landmarks to:* [`offline_pipeline/data/step6_bio_sanity/landmarks_bio.npy`](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/data/step6_bio_sanity/)

### Step 7: Tree Assignment

Matches each landmark to the closest tree from the tree coordinate map.

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
cd ~/ros2_ws/src/oskar/oskar_mapping/offline_pipeline
python3 step7_tree_assignment.py --max-frames -1
```

*Outputs tree counts to:* [`offline_pipeline/data/step7_tree_assignment/per_tree_flowers.json`](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/data/step7_tree_assignment/)

### Step 8: Thinning Decision

Calculates targets, removal priorities, and writes the final thinning plan.

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
cd ~/ros2_ws/src/oskar/oskar_mapping/offline_pipeline
python3 step8_thinning_decision.py --max-frames -1
```

*Outputs final plan to:* [`offline_pipeline/data/step8_thinning_decision/thinning_plan.json`](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/data/step8_thinning_decision/thinning_plan.json)

---

## 3. Diagnostic & Optimization Utilities

To analyze and fine-tune pipeline behavior without running the entire perception stack, we have implemented the following offline diagnostic scripts:

### Utility 1: Canopy Assignment Radius Sweep
Sweeps flower-to-tree assignment radii (`max_radius`) on your archived bio landmarks, compiling overlap counts, orphan rates, and scanned trees statistics.

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
cd ~/ros2_ws/src/oskar/oskar_mapping/offline_pipeline
python3 helpers/sweep_radius.py
```

*Outputs results report to:* [`docs/radius_sweep_report.md`](file:///home/workstation/ros2_ws/src/oskar/docs/radius_sweep_report.md) and [`data_archive/radius_sweep_2026-07-02/radius_sweep_results.json`](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/data/data_archive/radius_sweep_2026-07-02/radius_sweep_results.json).

### Utility 2: Dynamic Decimation Testing
Simulates processing decimated camera frames (e.g. taking every N-th frame) on your generated world observations, and calculates the resulting flower count accuracy drop.

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
cd ~/ros2_ws/src/oskar/oskar_mapping/offline_pipeline
# Test decimation factors N=1, 2, 3, 5
python3 helpers/test_decimation.py
```

*Outputs individual runs to:* `data/data_archive/decim_test_N{N}_2026-07-02/`
