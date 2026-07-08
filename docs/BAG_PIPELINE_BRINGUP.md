# OSKAR Phase 1 — Bag-Data Pipeline Bring-Up Log

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

> Living document. Records what we ran, what we assumed, and the actual results while
> bringing up the Phase 1 perception pipeline on the **real Fraunhofer apple-orchard bag**
> (not Isaac Sim). Updated as we make progress.
>
> Started 2026-06-09 · Last updated 2026-06-11

---

## 1. Goal & approach

Bring up the Phase 1 mapping pipeline (`oskar_mapping`) on the **real-world Fraunhofer IFAM
dataset** instead of Isaac Sim, node by node, validating each stage before adding the next.

Pipeline (per the design diagram):
```
Basler stereo → stereo_sync → disparity (depth) ┐
                            → segmentation (masks)┴→ depth_fusion → 3D flowers (cam frame)
   → backprojection (world) → multiview_fusion → bio_sanity → tree_assignment → thinning_decision
```

## 2. Dataset

Fraunhofer IFAM "Multisensor Dataset of Elstar Moje-Feindt Apple Trees During Bloom Stage."
Altes Land, N. Germany, recorded 2024-04-15, ~8.5 min, 238 trees, both sides traversed @
3.5 km/h. Location: `oskar_mapping/bagfile_data/`.

**Key structure discoveries:**
- The **stereo images are NOT in the rosbag.** They are JPGs in two zips, extracted to
  `bagfile_data/images/`:
  - `...SAMSON3...` = **TOP** camera (= the LEFT camera, see §6 left/right fix)
  - `...SAMSON4...` = **BOTTOM** camera (= the RIGHT camera)
  - **2566 matched pairs**, full res **4608×5328**, filenames `{id}_{epoch_sec}-{ms}.jpg`.
- The `.bag` files are **ROS 1** (read via `rosbags`), containing only:
  `/imu/data` (Imu, 256222), `/ublox/fix` (NavSatFix, 5124), `/gpsfix` (GPSFix, 5124),
  (+ `/ouster/points` LiDAR in the full bag).
- **No camera calibration is shipped with the dataset**, but a colleague provided SAMSON3/SAMSON4
  stereo calibration files later (see §11a). The feeder now uses those by default.
- Tree locations: `bagfile_data/A 27 Elstar Moje.geojson` → Field:1, Row:1, **Tree:238**.

## 3. Environment setup

ROS 2 Jazzy (Python 3.12) / ROS 2 Humble (Python 3.10). Workspace `~/ros2_ws`, repo in `src/oskar`.

**Build (only the oskar packages — plain `colcon build` fails on duplicate names elsewhere):**
```bash
cd ~/ros2_ws
colcon build --base-paths src/oskar --packages-select oskar_msgs oskar_mapping
```

**Isolated venv `~/oskar_venv`** — required because ROS `cv_bridge` needs **NumPy < 2** but
the global env has NumPy 2 (for `pyzed`/ZED). Created with `--system-site-packages` via `uv` (on Python 3.12) or standard `venv` (on Python 3.10);
contains numpy 1.26.4, opencv 4.11, FFM deps (`einops scipy scikit-image imageio pyyaml`),
and **detectron2 0.6** (CPU-only build — see §6).

**One-line env setup per terminal** (`oskar_env.sh`): sources ROS + workspace + activates venv:
```bash
source ~/ros2_ws/src/oskar/oskar_env.sh
```

**Run rule:** launch nodes as `python3 <node_file>.py` (NOT `ros2 run`), because the
installed `ros2 run` scripts hardcode the system python and bypass the venv → segfault.

## 4. Helper utilities added

- `oskar_mapping/oskar_mapping/bag_image_feeder.py` — reads the JPG pairs, publishes them as
  `Image` on `/rgb_static_1` (TOP) + `/rgb_static_2` (BOTTOM), plus real SAMSON3/SAMSON4
  CameraInfo on `/static_camera_info_1/2` by default. It still has a synthetic fallback if
  calibration files are missing. Params: `top_dir, bottom_dir, scale, rate_hz, loop,
  start_index, max_frames, left_calib_file, right_calib_file`. Run directly with `python3`.

## 5. Node-by-node results

### stereo_sync_node — ✅ VERIFIED
- Run: `--params-file mapping_params_sim.yaml -p use_sim_time:=false` (+ left/right swap, §6).
- Result: `/stereo/sync_pair` @ ~2 Hz. `ros2 topic echo --no-arr` showed a valid StereoPair —
  left+right images **identical timestamp** (sync correct), both 1332×1152 bgr8, both
  camera_infos attached. Logs "Images already rectified — skipping remap" (identity calib).

### disparity_node (Fast-FoundationStereo) — ✅ VERIFIED
- Run: `--params-file ... -p baseline_m:=0.13 -p max_disp:=384 -p tile_height:=768
  -p tile_overlap:=96 -p tile_width:=1024 -p tile_x_overlap:=384` for full-resolution tiled inference.
- FFM at `/home/workstation/Fast-FoundationStereo` (weights at configured path). No flash_attn
  needed (uses torch SDPA).
- Result: `/stereo/depth` (+ `/stereo/disparity`, `/stereo/points`) publishes a stitched
  full-resolution disparity/depth image. Point cloud in RViz shows **coherent 3D orchard
  structure** after the left/right fix. 2D rectangular tiling avoids the full-frame CUDA OOM and cuDNN errors while
  keeping the image at `scale:=1.0`.
- Rate: Running 72 tiles ($8 \times 9$) sequentially per frame yields an average rate of **~0.013 – 0.017 Hz** (roughly 60–90 seconds per frame). This is the expected performance trade-off for full-resolution processing on this FFM model checkpoint.


### segmentation_node (Mask R-CNN / detectron2) — ✅ VERIFIED
- Weights: `…/synthetic_apple_flowers/results/exp_bhangale/train/model_final.pth`
  (+ `detectron2_config.yaml` beside it; GeneralizedRCNN, 1 class = flower).
- Run: `-p model_path:=<above> -p confidence_threshold:=0.3 -p device:=cuda`.
- Standalone test: **37 flowers** on a full-res image in 1.26 s (GPU).
- In-pipeline: `/flowers/masks` published with e.g. **42 flowers** per (downscaled) frame.

### depth_fusion_node — ✅ VERIFIED
- Run: `-p left_info_topic:=/static_camera_info_1` (config default `/rgb_camera_1/camera_info`
  is wrong for the bag). **Timing requirement:** disparity (~0.5 Hz) and segmentation
  (~1 Hz) must process the SAME frame for the time-sync to match → **lower feeder to
  ~0.3 Hz** so both finish each frame.
- Result: `/flowers/detections_3d` — flowers with 3D `position` (camera frame). Example
  cluster z≈2.5–4 (units), x/y≈±1. The `low_confidence`/`depth_std_ratio` filter works
  (clean flowers std≈0.005–0.02 → `false`; noisy → `true`). One garbage outlier seen
  (z≈24175 from near-zero disparity) — to be rejected by later fusion/sanity stages.

### gnss_pose_tf_node — ✅ VERIFIED (robot pose → TF)
- Built by us (§9b). Loads `pose_trajectory.npz`, interpolates pose at each frame stamp,
  publishes TF `map→base_link` (+ static `base_link→static_camera_1`). Replaces the missing
  "RTK GPS + IMU fusion" box.

### backprojection_node — ✅ TESTED (camera 3D → world)
- Run: `--params-file ... -p use_sim_time:=false`. Publishes `/flowers/world_obs` (flowers in
  the map frame). Output is structurally plausible (z≈2 m heights, tight clusters) but lands
  **~5–8 m off the trees** — calibration gap, see §9c. **Per-frame perception is now tested
  end-to-end from images all the way to world-frame flowers.**

### multiview_fusion_node — ✅ TESTED (DBSCAN merge across frames)
- Run: `--params-file ... -p use_sim_time:=false`. Publishes `/flowers/landmarks`. Verified a
  merged landmark with `observation_count: 3` (same flower fused across 3 frames) with the
  default `eps=0.08 m`. Confirms the world-frame offset is **consistent frame-to-frame**, so
  fusion works even before calibration (calibration only shifts the cluster to the right place).
  NOTE: `eps=0.08` is too tight for the current noisy/non-metric positions → output very
  sparse (mostly empty messages). Loosened to **`eps:=0.3`** to densify for testing — then it
  produced 3 merged landmarks (obs_count 34/10/3). `eps=0.3` over-merges (lumps nearby
  flowers); proper `eps` needs the real depth scale (calibration). eps/min_samples are
  clustering knobs, fair to tune (not calibration).

### bio_sanity_node — ✅ TESTED (infer occluded corymb members)
- Run: `--params-file ... -p use_sim_time:=false`. In `/flowers/landmarks` → out
  `/flowers/landmarks_bio`. Empty-in→empty-out (so it looked empty until fusion was densified).
  With 3 confirmed landmarks in, it inflated each to a 5-flower corymb (1 real + 4
  `is_inferred:true` at obs_count 0, conf 0.3, in a ~1.5 cm ring) → 15 landmarks out. Working.

### tree_assignment_node — ✅ TESTED
- Run with the REAL tree map: `-p tree_map_file:=config/tree_map_real.yaml` (238 trees in ENU,
  generated from the geojson via pose_trajectory.npz) and `-p max_radius_m:=8.0` (flowers are
  ~6 m off, so the proper 0.8 m assigns nothing pre-calibration). Caught a frame with one tree
  at `confirmed_count: 6` — assignment logic fires (wrong tree due to offset, but proves plumbing).

### thinning_decision_node — ✅ TESTED (last node)
- Run: `--params-file ... -p use_sim_time:=false`. Publishes `/orchard/thinning_plan` (238-tree
  needs_thinning / rough_removal_count / priority_score). All `needs_thinning: false` on current
  data (per-tree counts below agronomic_max=25 — sparse + non-metric). Node works; real thinning
  flags need accurate counts (calibration).

### ✅ ALL 9 PHASE 1 NODES RUN END-TO-END on real bag data (2026-06-11).
World-frame accuracy still pending calibration (§9c); downstream counts are low/approximate.

## 6. Key fixes & diagnoses

- **NumPy 2 → cv_bridge segfault.** Global NumPy 2 (for pyzed) crashes ROS cv_bridge. Fixed
  with the isolated `~/oskar_venv` (numpy 1.26.4). Global env untouched.
- **Left/right cameras were swapped.** Measured the top↔bottom feature offset with ORB:
  **dx = −224 px, dy = 9.6 px** → horizontal baseline (good for FoundationStereo), but the
  sign meant TOP = left, BOTTOM = right. Pipeline was feeding them reversed → garbage "fan"
  cloud. Fix: `stereo_sync -p left_image_topic:=/rgb_static_1 -p right_image_topic:=/rgb_static_2`.
- **detectron2 build.** System CUDA toolkit is 12.8/13.0 (no 12.4 to match torch cu124) → the
  normal CUDA build fails at link. Fixed by building **CPU-only**:
  `CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0 pip install --no-build-isolation 'git+…/detectron2.git'`.
  Mask R-CNN still runs on GPU via torchvision ops.
- **max_disp scales with resolution, but is checkpoint-limited.** Measured shift ~224 px @ full
  res. Full-res needs more than the old 192, but this FFM checkpoint's positional embedding
  fails at 512; use `max_disp:=384`.

## 7. Global assumptions & caveats

1. **Real CameraInfo by default** — feeder loads `SAMSON3_SAMSON4_stereo.yaml` for TOP/left and
   `SAMSON4_SAMSON3_stereo.yaml` for BOTTOM/right. Synthetic CameraInfo remains only as fallback
   if files are missing.
2. **Full-resolution images** — current feeder run uses `scale = 1.0`.
3. **Disparity tiling** — full 5328x4608 frames are processed using 2D rectangular tiling
   (`tile_height:=768`, `tile_overlap:=96`, `tile_width:=1024`, `tile_x_overlap:=384`) and stitched back into one disparity image. This avoids cuDNN dimension limits.
4. Left = TOP (`/rgb_static_1`), Right = BOTTOM (`/rgb_static_2`).
5. World-frame accuracy still depends on camera-to-platform mounting calibration.

## 8. Current state & what's proven

**Proven end-to-end on real data:** images → synced → depth, images → flower masks,
depth + masks → **flowers in 3D (camera frame)**. The full per-frame perception chain works.

**Update:** the pose node is now built and backprojection is tested (see §5, §9b). The
per-frame chain works end-to-end into the world frame. **Biggest remaining gap is now
CALIBRATION** (§9c) — world flowers are ~5–8 m off the trees until real stereo + mounting
calibration replaces our placeholders. Downstream nodes (multiview_fusion onward) wait on it.

## 9. Pose node — feasibility (confirmed 2026-06-11)

- **IMU has absolute orientation** (`orientation_covariance[0]=0.01`, not −1) → no orientation
  estimation needed.
- **RTK position** `/ublox/fix` status=2, ~1.4 cm. `/gpsfix` has **no heading** (track sentinel)
  → heading comes from the IMU.
- **Clocks align:** camera JPG epochs and bag ROS times overlap 512.3 s of ~513 s → pose can
  be interpolated at each photo's timestamp.
- **Tree map:** 238 Point features parse from the geojson.
- **ENU check:** robot track E[−115,114] N[−33,33] path ≈490 m (both sides); trees
  E[−109,107] N[−30,30] — robot drove alongside the trees. Geometry valid.

### 9a. IMU heading check (2026-06-11) — IMU YAW IS UNUSABLE (drifts)

Compared IMU yaw to GNSS course-over-ground:
- GNSS heading is clean & bimodal (≈165° / ≈345° = down the row and back).
- IMU yaw vs GNSS: circ-std 64.8° overall — poor. Per-time-slice analysis shows it's NOT
  random: the offset is two **stable plateaus** (≈−33° for 0–256s, ≈−147° for 256–512s) with
  a sudden **~113° STEP at the 256s midpoint = the U-turn**. A correct IMU would hold one
  constant offset across the turn. The IMU mis-tracks the U-turn rotation and never recovers
  ⇒ **gyro-only yaw, NO magnetometer to re-anchor heading**. Roll/pitch are fine
  (gravity-referenced); body +z ~177° from up (mounted z-down, near level, ±6°).

**Conclusion: do NOT use IMU orientation for heading.** Derive heading from GNSS instead.

### 9b. Revised pose plan
```
position  = RTK GNSS → ENU                      (1.4 cm)
heading   = GNSS course-over-ground (smoothed)  (NOT the IMU; clean, robot moves 91% of time)
roll/pitch= from IMU (small ~6°) or assume 0
```
Pose node: load GNSS track → ENU; compute per-sample heading from motion (hold last good
heading when stationary/turning at row ends); interpolate position+heading at each camera
frame stamp; publish TF `map→base_link` + static `base_link→static_camera_1`. The camera
extrinsic carries a ~90° offset (cameras point sideways at trees) + downward tilt — start
with a guess, refine by checking projected flowers land on the geojson tree points.

## 9c. Backprojection result + DATA NEEDED FROM SUPERVISOR (2026-06-11)

Backprojection runs: `/flowers/world_obs` puts flowers in the map frame at sensible heights
(z≈2 m), tightly clustered. BUT they land **~5–8 m off the trees** (median 5.5 m). Diagnosis
on one frame: robot position is correct (2 m from the row — RTK good), but (1) the camera is
pointing at the **wrong side** (trees were on the robot's left; mount guess said right), and
(2) the **depth scale is ~3× too large** (flowers placed ~6 m out vs true ~2 m — non-metric).

**DECISION (user):** do NOT fit/auto-tune these to make flowers land on trees — that is
overfitting and hides the missing calibration. Keep them as labeled assumptions; get the
**real calibration from the supervisor.** (See memory `no-fitting-calibration-to-data`.)

**Calibration data to request from supervisor (replaces our placeholders):**
1. **Stereo intrinsics** — K (fx, fy, cx, cy) + distortion for EACH Basler camera.
   (placeholder now: fx = 0.9·width, zero distortion → depth non-metric)
2. **Stereo extrinsics** — relative rotation + translation between the two cameras
   (baseline ≈ 0.13 m). Needed for rectification AND metric depth.
   (placeholder now: identity rectification, baseline_m = 0.13)
3. **Camera→platform mounting** — pose of the camera(s) w.r.t. base_link / GNSS antenna / IMU:
   which side they face, position, orientation.
   (confirmed: gnss_pose_tf_node `camera_side=left` (confirmed as main camera side by user), xyz=[0.1,0,1.0], level)
4. (minor) GNSS antenna lever-arm w.r.t. base_link.

Until these arrive, world-frame output is structurally plausible but **not metrically located**.

> **Startup commands:** see [RUN_PIPELINE.md](RUN_PIPELINE.md) for the full copy-paste runbook.

## 11. 2026-06-18 session — real calibration + infrastructure fixes

### 11a. Real stereo calibration arrived (from a colleague)
A colleague shared the real stereo calibration for the same rig:
`bagfile_data/SAMSON3_SAMSON4_stereo.yaml` (left/reference, P[0,3]=0) and
`SAMSON4_SAMSON3_stereo.yaml` (right, P[0,3]=−394.1). Contains real **cameraMatrix**
(fx≈3030, fy≈3029, cx≈2644, cy≈2280 @ full 5328×4608), **distortion**, **rotation**, and
**projectionMatrix**. Baseline derived from P: **0.1302 m** — matches the README's "~13 cm",
strong evidence it's the same physical rig. (Web search confirmed: no calibration is published
with the dataset; Basler ace 2 sensor = IMX183, pixel pitch 2.74 µm.)

- **Feeder uses it by default:** params `left_calib_file` / `right_calib_file` default to the
  bundled SAMSON3/SAMSON4 YAMLs. The feeder publishes real K/D/R/P scaled to the emitted image
  size (D, R are scale-invariant; K, P scale by diag(sx,sy,1)); `stereo_sync` then does a real
  undistort+rectify. If the files are missing, it falls back to synthetic CameraInfo.
- **Status:** real stereo calibration is wired by default and `stereo_sync` builds rectification
  maps. Metric end-to-end re-test is pending. **Confirm with supervisor/colleague that this
  calibration was made with the SAME cameras+lenses that recorded our bag** before trusting
  final metric output.

### 11c. Full-resolution disparity tiling (2026-06-30)
The user wants to keep full-resolution input (`scale:=1.0`) instead of downscaling. Full
5328x4608 FFM inference OOMs on the available GPU, so `disparity_node` gained optional
horizontal-band inference:

```bash
-p max_disp:=384 -p tile_height:=768 -p tile_overlap:=96
```

Important: the image is cut into **horizontal bands**, not vertical tiles. Stereo matching searches
horizontally along each row, so each band keeps the full width. With height 4608, tile height 768,
and overlap 96, the node processes about 8 overlapping bands and stitches the center rows back
into one full-size disparity/depth image. Use `max_disp:=384`; `max_disp:=512` triggers a
positional-embedding shape error with the current checkpoint. If GPU memory is still tight,
reduce `tile_height` to 512; if visible band boundaries appear, increase `tile_overlap` to 128.
Each tile is copied into contiguous memory before inference because cuDNN can reject sliced NumPy
views with `CUDNN_STATUS_NOT_SUPPORTED`.

### 11b. Infrastructure fixes to get an end-to-end run (NOT calibration related)
After a reboot the pipeline stopped flowing past segmentation. Root causes, in order of impact:

1. **QoS — large topics need RELIABLE (the real blocker).** `/stereo/sync_pair` (~9 MB) and
   `/flowers/masks` (tens of MB — a full-res mask per flower) were published fine but their
   **subscribers used BEST_EFFORT**, which silently drops large fragmented messages on
   localhost → disparity starved, depth_fusion's time-sync never paired depth+masks → 0
   detections. **Proof:** `ros2 topic hz /stereo/sync_pair` (RELIABLE) read full 0.3 Hz while
   disparity's BEST_EFFORT sub starved. **Fix (applied, on branch):** flip to RELIABLE in
   `disparity_node` (sync_pair sub), `segmentation_node` (sync_pair sub + masks pub, masks
   pub uses depth=1), `depth_fusion_node` (depth/masks/info subs). `stereo_sync` pub was
   already RELIABLE. → `/flowers/detections_3d` then flowed at **0.3 Hz**. See memory
   `qos-reliable-for-large-msgs`.
2. **Isaac Sim starves the GPU.** A background Isaac Sim sim drove disparity to **0.003 Hz**
   (1 frame / several min). Close Isaac Sim before running the pipeline.
3. **GPU contention.** disparity + segmentation both default to GPU 0 and time-share → ~half
   speed. Put segmentation on GPU 1: `CUDA_VISIBLE_DEVICES=1 … device:=cuda`.
4. **venv must be sourced in EVERY terminal.** Missing `source oskar_env.sh` →
   `ModuleNotFoundError: detectron2` (segmentation) or NumPy-2/`cv_bridge` `_ARRAY_API`
   crash (depth_fusion). The venv has numpy 1.26.4 + detectron2.
5. **confidence_threshold 0.3, not 0.08.** 0.08 yields ~99 junk masks → ~150 MB message; 0.3
   gives ~30–40 → ~64 MB. (The proper long-term fix is to not embed a full-res mask per flower.)
6. **OS socket buffers** (`net.core.rmem_max`/`wmem_max`) reset small on reboot (ZED config
   sets rmem=1 MB; wmem defaults to 208 KB). Bumping both to 2 GB only *half*-helped — it was
   a red herring; **RELIABLE QoS (#1) was the actual cure.** Harmless to set, persist via
   `/etc/sysctl.d/` if desired.

**Result:** on a clean state (Isaac closed, stale `/dev/shm/fastrtps_*` cleared, venv sourced,
GPU split) + the RELIABLE QoS fix, the pipeline runs cleanly through **depth_fusion** —
`/flowers/detections_3d` at the full 0.3 Hz with real 3D flowers (some garbage-z outliers,
flagged by `depth_std_ratio`, are expected on synthetic calibration).

## 10. Changelog
- 2026-06-09: Started bag testing; built packages; wrote feeder; verified stereo_sync.
- 2026-06-11: venv/numpy fix; disparity (FFM) verified + left/right fix; detectron2 CPU build;
  segmentation verified; depth_fusion verified (flowers in 3D); pose feasibility confirmed.
- 2026-06-11: IMU heading rejected (U-turn step, no magnetometer) → heading from GNSS.
  Built **2a** `extract_pose_trajectory.py` (→ `bagfile_data/pose_trajectory.npz`: t,E,N,U,yaw
  + tree points in ENU; origin = tree centroid) — RAN OK (5124 poses, yaw bimodal ≈−13°/167°).
  Built **2b** `gnss_pose_tf_node.py` (interpolates pose at each /static_camera_info_1 stamp →
  TF map→base_link + static base_link→static_camera_1; camera mount guess = looks right,
  level; math validated: look-dir 90° from travel). NEXT: run 2b + verify TF, then wire
  backprojection and check flowers land on tree points (tune camera mount).
- 2026-06-11: backprojection tested (`/flowers/world_obs`, ~5–8 m off trees → calibration gap,
  §9c; decided NOT to fit calibration). multiview_fusion tested (eps loosened 0.08→0.3 to
  densify). bio_sanity tested (inflates corymbs to 5, `is_inferred` flowers). Remaining:
  tree_assignment, thinning_decision. Docs (PROJECT_STATUS, RUN_PIPELINE, node explainers) updated.
- 2026-06-11: tree_assignment tested (real 238-tree map `tree_map_real.yaml`, max_radius 8.0 →
  one tree got 6). thinning_decision tested (full 238-tree plan, all needs_thinning false on low
  counts). **ALL 9 Phase 1 nodes now run end-to-end on real data.** Wrote 9_thinning explainer;
  all node explainers carry change/why/result/assumption sections.
- 2026-06-18: real stereo calibration received from colleague (SAMSON3/4 `*_stereo.yaml`,
  baseline 0.1302 m); feeder gained `left_calib_file`/`right_calib_file` (synthetic fallback).
  Diagnosed a post-reboot stall: **large topics need RELIABLE QoS** (BEST_EFFORT subscribers
  drop 9 MB sync_pair / ~64 MB masks) — applied the QoS fix to disparity/segmentation/
  depth_fusion → `/flowers/detections_3d` flows at 0.3 Hz. Also: close Isaac Sim (GPU starve),
  split GPUs (`CUDA_VISIBLE_DEVICES=1` for segmentation), source venv per terminal,
  confidence 0.3. See §11 and memory `qos-reliable-for-large-msgs`.
