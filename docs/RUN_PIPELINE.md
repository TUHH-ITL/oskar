# Running the OSKAR Phase 1 Pipeline (real bag data)

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

Copy-paste runbook to bring the whole Phase 1 perception pipeline up from scratch on the
Fraunhofer orchard dataset. One node per terminal.

> **Every terminal must start with this line** (sources ROS + workspace + activates the venv):
>
> ```bash
> source ~/ros2_ws/src/oskar/oskar_env.sh
> ```
>
> Run nodes with `python3 /full/path/to/file.py` (NOT `ros2 run` — that bypasses the venv and segfaults).

---

## 0. One-time setup (already done; only redo on a fresh machine)

```bash
# build the two packages (plain `colcon build` fails on duplicate names elsewhere in src)
cd ~/ros2_ws
colcon build --base-paths src/oskar --packages-select oskar_msgs oskar_mapping

# python venv with numpy<2 (cv_bridge) + detectron2 + FFM deps
# OPTION A: For Ubuntu 24.04 / ROS 2 Jazzy (using uv):
uv venv ~/oskar_venv -p 3.12 --clear --system-site-packages
uv pip install -p ~/oskar_venv/bin/python3 'numpy<2' opencv-python einops scipy scikit-image imageio pyyaml matplotlib pandas openpyxl rosbags torch torchvision timm scikit-learn
CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0 uv pip install -p ~/oskar_venv/bin/python3 --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"

# OPTION B: For Ubuntu 22.04 / ROS 2 Humble (using standard venv):
# python3 -m venv ~/oskar_venv --system-site-packages
# ~/oskar_venv/bin/pip install 'numpy<2' opencv-python einops scipy scikit-image imageio pyyaml timm scikit-learn
# CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0 ~/oskar_venv/bin/pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# extract the camera images from the two zips into bagfile_data/images/
cd ~/ros2_ws/src/oskar/oskar_mapping/bagfile_data
unzip -q -o 2024-04-15_10-59-41_A27_Bluete_top_camera_SAMSON3.zip -d images/
unzip -q -o 2024-04-15_10-59-41_A27_Bluete_botom_camera_SAMSON4.zip -d images/

# build the GNSS pose trajectory file (bagfile_data/pose_trajectory.npz)
cd ~/ros2_ws/src/oskar/oskar_mapping/oskar_mapping
~/oskar_venv/bin/python ~/ros2_ws/src/oskar/oskar_mapping/oskar_mapping/extract_pose_trajectory.py

# generate the real 238-tree map (config/tree_map_real.yaml) from that trajectory
~/oskar_venv/bin/python - <<'PY'
import numpy as np, os
d = np.load(os.path.expanduser("~/ros2_ws/src/oskar/oskar_mapping/bagfile_data/pose_trajectory.npz"))
out = os.path.expanduser("~/ros2_ws/src/oskar/oskar_mapping/config/tree_map_real.yaml")
with open(out, "w") as f:
    f.write("# 238 real orchard trees; x=ENU East, y=ENU North (same frame as /flowers/world_obs)\ntrees:\n")
    for i,(x,y) in enumerate(zip(d["tree_E"], d["tree_N"])):
        f.write(f"  {i}: {{x: {x:.3f}, y: {y:.3f}}}\n")
print("wrote", out)
PY
```

---

## Startup order — one terminal each

Image folder paths (used by the feeder):

- TOP  = `…/bagfile_data/images/2024-04-15_10-59-41_A27_Bluete_SAMSON3_1713171581`
- BOT  = `…/bagfile_data/images/2024-04-15_10-59-41_A27_Bluete_SAMSON4_1713171581`

### Terminal 1 — image feeder  (→ /rgb_static_1, /rgb_static_2, /static_camera_info_*)

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
python3 ~/ros2_ws/src/oskar/oskar_mapping/oskar_mapping/bag_image_feeder.py --ros-args \
  -p top_dir:=$HOME/ros2_ws/src/oskar/oskar_mapping/bagfile_data/images/2024-04-15_10-59-41_A27_Bluete_SAMSON3_1713171581 \
  -p bottom_dir:=$HOME/ros2_ws/src/oskar/oskar_mapping/bagfile_data/images/2024-04-15_10-59-41_A27_Bluete_SAMSON4_1713171581 \
  -p scale:=1.0 -p rate_hz:=0.3 -p loop:=true
```

> `rate_hz:=0.3` so disparity (~0.5 Hz) and segmentation (~1 Hz) both finish each frame
> (required for depth_fusion's time-sync). Use `2.0` if running only the front nodes.
>
> **Continuous playback:** add `-p loop:=true` to restart from frame 0 after the last of the
> 2566 pairs (otherwise the feeder stops after one pass and the pipeline goes idle). Use
> `-p start_index:=N` / `-p max_frames:=N` to play only a range.

### Terminal 2 — stereo_sync  (→ /stereo/sync_pair)   NOTE the left/right swap

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
python3 ~/ros2_ws/src/oskar/oskar_mapping/oskar_mapping/stereo_sync_node.py --ros-args \
  --params-file ~/ros2_ws/src/oskar/oskar_mapping/config/mapping_params_sim.yaml \
  -p use_sim_time:=false \
  -p left_image_topic:=/rgb_static_1 -p right_image_topic:=/rgb_static_2
```

### Terminal 3 — segmentation (Mask R-CNN, **GPU 1**)  (→ /flowers/masks)

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
CUDA_VISIBLE_DEVICES=1 python3 ~/ros2_ws/src/oskar/oskar_mapping/oskar_mapping/segmentation_node.py --ros-args \
  --params-file ~/ros2_ws/src/oskar/oskar_mapping/config/mapping_params_sim.yaml \
  -p use_sim_time:=false \
  -p model_path:=/home/workstation/oskar/synthetic_apple_flowers/results/exp_bhangale/train/model_final.pth \
  -p confidence_threshold:=0.3 -p device:=cuda
```

> `CUDA_VISIBLE_DEVICES=1` runs segmentation on GPU 1 so it doesn't contend with disparity
> (GPU 0). Keep `confidence_threshold:=0.3` — lower (e.g. 0.08) floods the masks message with
> ~99 junk detections (~150 MB) and makes delivery worse.

### Terminal 4 — Disparity Estimation (Choose Option A or B)

#### Option A: Dense Full-Frame Disparity (Tiled) (→ /stereo/depth, /stereo/points)

Processes the entire image using overlapping horizontal bands (takes ~60–90 seconds per frame, average rate ~0.015 Hz).

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
python3 ~/ros2_ws/src/oskar/oskar_mapping/oskar_mapping/disparity_node.py --ros-args \
  --params-file ~/ros2_ws/src/oskar/oskar_mapping/config/mapping_params_sim.yaml \
  -p use_sim_time:=false -p baseline_m:=0.13 -p max_disp:=384 \
  -p tile_height:=768 -p tile_overlap:=96 \
  -p tile_width:=1024 -p tile_x_overlap:=384
```

#### Option B: Sparse ROI-Based Disparity (Fast) (→ /stereo/depth, /stereo/points)

Processes FFM only on cropped flower regions detected by the segmentation node (takes ~1.5–3 seconds per frame, matching the full 0.3 Hz feeder rate).

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
python3 ~/ros2_ws/src/oskar/oskar_mapping/oskar_mapping/disparity_roi_node.py --ros-args \
  --params-file ~/ros2_ws/src/oskar/oskar_mapping/config/mapping_params_sim.yaml \
  -p use_sim_time:=false -p baseline_m:=0.13 -p max_disp:=384
```

### Terminal 5 — depth_fusion  (→ /flowers/detections_3d, camera frame)

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
python3 ~/ros2_ws/src/oskar/oskar_mapping/oskar_mapping/depth_fusion_node.py --ros-args \
  --params-file ~/ros2_ws/src/oskar/oskar_mapping/config/mapping_params_sim.yaml \
  -p use_sim_time:=false \
  -p left_info_topic:=/static_camera_info_1
```

### Terminal 6 — GNSS pose → TF  (→ map→base_link, base_link→static_camera_1)

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
python3 ~/ros2_ws/src/oskar/oskar_mapping/oskar_mapping/gnss_pose_tf_node.py
```

### Terminal 7 — backprojection  (→ /flowers/world_obs, world/map frame)

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
python3 ~/ros2_ws/src/oskar/oskar_mapping/oskar_mapping/backprojection_node.py --ros-args \
  --params-file ~/ros2_ws/src/oskar/oskar_mapping/config/mapping_params_sim.yaml \
  -p use_sim_time:=false
```

### Terminal 8 — multiview_fusion (DBSCAN merge across frames)  (→ /flowers/landmarks)

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
python3 ~/ros2_ws/src/oskar/oskar_mapping/oskar_mapping/multiview_fusion_node.py --ros-args \
  --params-file ~/ros2_ws/src/oskar/oskar_mapping/config/mapping_params_sim.yaml \
  -p use_sim_time:=false -p eps:=0.3 -p max_age_s:=300.0

```

> `eps:=0.3` (vs default 0.08): the non-metric/noisy positions scatter >8 cm, so 0.08 almost
> never clusters (stream stays empty). 0.3 makes clusters form but **over-merges** — it's a TEST
> value; the correct eps (~cm) needs metric calibration. Clustering knob, not calibration.

### Terminal 9 — bio_sanity (infer occluded corymb flowers)  (→ /flowers/landmarks_bio)

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
python3 ~/ros2_ws/src/oskar/oskar_mapping/oskar_mapping/bio_sanity_node.py --ros-args \
  --params-file ~/ros2_ws/src/oskar/oskar_mapping/config/mapping_params_sim.yaml \
  -p use_sim_time:=false
```

### Terminal 10 — tree_assignment (assign flowers to trees)  (→ /orchard/per_tree_flowers)

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
python3 ~/ros2_ws/src/oskar/oskar_mapping/oskar_mapping/tree_assignment_node.py --ros-args \
  --params-file ~/ros2_ws/src/oskar/oskar_mapping/config/mapping_params_sim.yaml \
  -p use_sim_time:=false \
  -p tree_map_file:=$HOME/ros2_ws/src/oskar/oskar_mapping/config/tree_map_real.yaml \
  -p max_radius_m:=8.0
```

> Uses the REAL 238-tree map. `max_radius_m:=8.0` is a TEST value — flowers are ~6 m off the
> trees, so the proper 0.8 m assigns nothing until calibration.

### Terminal 11 — thinning_decision (per-tree thinning plan)  (→ /orchard/thinning_plan)

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
python3 ~/ros2_ws/src/oskar/oskar_mapping/oskar_mapping/thinning_decision_node.py --ros-args \
  --params-file ~/ros2_ws/src/oskar/oskar_mapping/config/mapping_params_sim.yaml \
  -p use_sim_time:=false
```

---

## Verify (any extra terminal)

```bash
source ~/ros2_ws/src/oskar/oskar_env.sh

ros2 topic hz /stereo/sync_pair          # ~feeder rate
ros2 topic hz /stereo/depth              # ~0.5 Hz (FoundationStereo)
ros2 topic echo /flowers/masks --no-arr --once     # 'flowers: length: N'
ros2 topic echo /flowers/detections_3d --once      # 3D flowers (camera frame)
ros2 run tf2_ros tf2_echo map base_link            # robot pose moving through ENU
ros2 topic echo /flowers/world_obs --once          # flowers in world/map frame

# mid-pipeline topics are SPARSE — use grep to catch a populated frame (NOT --once, which
# usually catches an empty message):
ros2 topic echo /flowers/landmarks --field landmarks | grep -m1 landmark_id          # multiview
ros2 topic echo /flowers/landmarks_bio --field landmarks | grep -m1 landmark_id      # bio_sanity
ros2 topic echo /orchard/per_tree_flowers --field confirmed_counts | grep -m1 '[1-9]'  # assignment
ros2 topic echo /orchard/thinning_plan --field needs_thinning --once    # 238 bools (always populated)
```

RViz (fixed frame = `map`): add **TF**, **PointCloud2** on `/stereo/points`.
(No manual static_transform_publisher needed anymore — Terminal 6 publishes the TF tree.)

---

## Notes / gotchas

- Nodes can be started/stopped independently while the rest keep running (ROS 2 is dynamic).
- Order isn't strict, but downstream nodes idle (or warn) until their input appears.
- Feeder publishes real SAMSON3/SAMSON4 stereo CameraInfo by default. World-frame accuracy still
  depends on camera-to-platform mounting calibration.
- Left = `/rgb_static_1` (TOP camera), Right = `/rgb_static_2` (BOTTOM). Do NOT swap back.
- **Mid-pipeline topics (`/flowers/landmarks`, `_bio`, `/orchard/per_tree_flowers`) are sparse**
  — most messages are empty because multiview only clusters intermittently. Use the `grep`
  catches above, not `--once`.
- `-p eps:=0.3` (Terminal 8) and `-p max_radius_m:=8.0` (Terminal 10) are **TEST values** to make
  the plumbing visible despite the ~6 m calibration offset. With real calibration, use the proper
  tight `eps` (~0.08) and `max_radius` (0.8). These are algorithm knobs, NOT calibration.
- See `BAG_PIPELINE_BRINGUP.md` for the full bring-up log, assumptions, and results.

### Real stereo calibration (feeder)

The feeder uses the rig's bundled stereo calibration by default:
`bagfile_data/SAMSON3_SAMSON4_stereo.yaml` for TOP/left and
`bagfile_data/SAMSON4_SAMSON3_stereo.yaml` for BOTTOM/right.

To override those defaults, add explicit files to **Terminal 1**:

```bash
  -p left_calib_file:=$HOME/ros2_ws/src/oskar/oskar_mapping/bagfile_data/SAMSON3_SAMSON4_stereo.yaml \
  -p right_calib_file:=$HOME/ros2_ws/src/oskar/oskar_mapping/bagfile_data/SAMSON4_SAMSON3_stereo.yaml
```

Feeder logs `Calibration: left=real ... right=real ...`; stereo_sync then does a real
undistort+rectify (baseline 0.1302 m). Confirm provenance with supervisor before trusting final
metric output.

---

## Troubleshooting (learned the hard way — 2026-06-18)

If the pipeline stalls or `detections_3d` stays empty, check these **in order**:

1. **`source ~/ros2_ws/src/oskar/oskar_env.sh` in EVERY terminal.** Missing it →
   `ModuleNotFoundError: detectron2` (segmentation) or NumPy-2/`cv_bridge` `_ARRAY_API` crash
   (depth_fusion). The prompt must read `(oskar_venv) …`.
2. **Close Isaac Sim.** A running Isaac sim starves the GPU → disparity drops to ~0.003 Hz
   (a frame every several minutes). `nvidia-smi` shows an `IsaacLab`/`kit` process if it's up.
3. **Split GPUs.** Run segmentation with `CUDA_VISIBLE_DEVICES=1` (Terminal 4) so it doesn't
   share GPU 0 with disparity.
4. **QoS must be RELIABLE on the big topics** (already fixed in the nodes). If `detections_3d`
   is empty but `ros2 topic hz /flowers/detections_3d` works after the fix, you're good. The
   large topics (`sync_pair` ~9 MB, `masks` ~64 MB) were dropped by BEST_EFFORT subscribers;
   `disparity`/`segmentation`/`depth_fusion` subscribers are now RELIABLE. See
   `BAG_PIPELINE_BRINGUP.md` §11.
5. **Stale DDS state after many restarts.** With all nodes stopped:
   `rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_*` then relaunch.
6. **Measuring rates:** use `ros2 topic hz` **directly in a terminal** (not piped). It's
   unreliable for the huge `masks`/`sync_pair` messages (use `echo` to confirm those flow);
   it's accurate for small ones (`detections_3d`, `world_obs`, `landmarks`).
7. **(optional) OS buffers:** `sudo sysctl -w net.core.rmem_max=2147483647 net.core.wmem_max=2147483647`
   — a reboot resets these small. Secondary to the QoS fix, but harmless.
