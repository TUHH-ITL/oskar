# OSKAR — Project Status

> High-level "where are we" doc. Read this first, then:
> - [BAG_PIPELINE_BRINGUP.md](BAG_PIPELINE_BRINGUP.md) — detailed bring-up log.
> - [RUN_PIPELINE.md](RUN_PIPELINE.md) — copy-paste commands to launch the online ROS 2 pipeline.
> - [RUN_OFFLINE_PIPELINE.md](RUN_OFFLINE_PIPELINE.md) — instructions to run the high-performance offline pipeline.
> - [3_disparity_node_explained.md](3_disparity_node_explained.md) / [3_1_disparity_roi_node_explained.md](3_1_disparity_roi_node_explained.md) — deep dive on depth nodes.
>
> Last updated: 2026-07-01

---

## 1. What this project is

**OSKAR** — an autonomous apple-orchard robot (Tipard base + UR20 arm + Robotiq gripper)
that drives orchard rows, builds a 3D map of flowers per tree (**Phase 1: map**), then
prunes/thins excess flowers (**Phase 2: prune**). Runs on **ROS 2 Humble**.

We are bringing up **Phase 1 on a real-world ROS bag** (Fraunhofer IFAM orchard dataset),
not Isaac Sim. **Phase 2 is not written** in this repo (an `oskar_pruning` package appears in
`ros2 pkg list` but isn't in the git tree — worth locating later).

## 2. Phase 1 pipeline — TEST STATUS (on real bag data)

```
[feeder]          JPG stereo pairs -> ROS images + real camera_info    ✅ tested
   │
   ├─► [segmentation]  Mask R-CNN flower masks  -> /flowers/masks       ✅ tested (detectron2)
   ▼
[stereo_sync]     sync + (identity) rectify     -> /stereo/sync_pair    ✅ tested
   ▼
[disparity]       Fast-FoundationStereo depth   -> /stereo/depth        ✅ tested
   ▼
[depth_fusion]    depth under each mask, pixel->3D (cam frame)          ✅ tested
   ▼                                  ▲ robot pose (TF)
[backprojection]  3D cam -> 3D world  └─ [gnss_pose_tf]  GNSS->TF        ✅ tested (both)
   ▼                                       (heading from GNSS, see §5)
[multiview_fusion] merge across frames (DBSCAN)                          ✅ tested
   ▼
[bio_sanity]       infer occluded flowers per corymb                    ✅ tested
   ▼
[tree_assignment]  assign flowers to nearest known tree                 ✅ tested
   ▼
[thinning_decision] count vs target -> /orchard/thinning_plan           ✅ tested
```

**Tested end-to-end from images all the way through 3D-backprojection into the world frame.**
World-frame flowers are produced (`/flowers/world_obs`) but are **not metrically accurate**
yet — see §6 (calibration). **All 9 Phase 1 boxes now run end-to-end on real bag data**
(world-frame accuracy still pending calibration).

## 3. Repo layout

| Package | Purpose | State |
|---|---|---|
| `oskar_simulation/` | Isaac Sim scenes, URDF/meshes, MoveIt, teleop | Working (mature part) |
| `oskar_msgs/` | 11 custom messages | Done |
| `oskar_mapping/` | Phase 1 perception (9 nodes + our helpers) | **Built & run through backprojection on bag data** |
| `oskar_pruning/` | Phase 2 | Not in repo |

**Helper scripts we added** (in `oskar_mapping/oskar_mapping/`, run with `python3`):
- `bag_image_feeder.py` — publishes the dataset's JPG stereo pairs + real SAMSON3/SAMSON4
  camera_info by default, with synthetic fallback if calibration files are missing.
- `extract_pose_trajectory.py` — one-off: bag GNSS → `bagfile_data/pose_trajectory.npz`.
- `gnss_pose_tf_node.py` — publishes robot pose as TF from that trajectory.

## 4. Environment & build

- Workspace `~/ros2_ws`, repo in `src/oskar`. Other unrelated projects share `src/` and have
  duplicate package names, so a plain `colcon build` fails. Build oskar in isolation:
  ```bash
  cd ~/ros2_ws
  colcon build --base-paths src/oskar --packages-select oskar_msgs oskar_mapping
  ```
- **Isolated venv `~/oskar_venv`** (numpy<2 for cv_bridge; global env has numpy 2 for ZED).
  Contains opencv, FoundationStereo deps, and detectron2 (CPU-built). Activate everything
  per terminal with `source ~/ros2_ws/src/oskar/oskar_env.sh`.
- Run nodes as `python3 <file>` (NOT `ros2 run` — it bypasses the venv).
- Full setup + launch commands: **see RUN_PIPELINE.md**.

## 5. Key things we learned / decided

- **New Dataset Switch (Blossom2024 Sensorbox1)**: We transitioned the pipeline to process the 354-tree subset of Blossom2024. Left images are lossless PNG files (`tree_0001.png` to `tree_0354.png`) under `Basler_Left_Elstar/`. 
- **Raw REC Extraction & Rectification**: We extracted the matching 354 right camera images from the 59 GB raw `SAMSON4` `.rec` file. These were processed in horizontal space using the new `SAMSON4_SAMSON3_stereo.yaml` calibration, rectified silently, rotated 270 degrees CCW (matching the vertical sensor physical orientation), and saved as `tree_XXXX.png` to `Basler_Right_Elstar/`.
- **Dynamic Binary-Seek Timestamp Mapping**: The tree images are named sequentially (`tree_0001` to `tree_0354`) which mapped linearly to frame indices `114` to `467` in the raw camera streams. We updated the parser (`utils.parse_frame_timestamp`) to dynamically open `SAMSON3_1713171581.rec` and perform a sub-microsecond seek to retrieve exact hardware timestamps for each tree frame, keeping coordinate projection perfectly synchronized.
- **Cameras aren't in the bag** — they're JPGs in two zips (TOP=SAMSON3=left,
  BOTTOM=SAMSON4=right; 2566 pairs). The bag has only IMU/GNSS/LiDAR (ROS 1 format).
- **Real stereo calibration is now wired by default.** The feeder loads
  `bagfile_data/SAMSON3_SAMSON4_stereo.yaml` and `SAMSON4_SAMSON3_stereo.yaml`
  automatically (baseline 0.1302 m), publishes real CameraInfo, and `stereo_sync` builds
  rectification maps from it. Confirm provenance before trusting final metric results.
- **Full-resolution disparity now uses horizontal tiling.** Instead of downscaling the feeder,
  `disparity_node` can process full 5328x4608 frames as overlapping horizontal bands
  (`tile_height:=768`, `tile_overlap:=96`) and stitch them back into one disparity image (see [3_disparity_node_explained.md](3_disparity_node_explained.md)).
- **Alternative: ROI-based crop depth estimation.** For 20x to 30x faster inference, `disparity_roi_node` runs Mask R-CNN segmentation first, and computes stereo matching only on bounding boxes around detected flowers (see [3_1_disparity_roi_node_explained.md](3_1_disparity_roi_node_explained.md)).
- **High-Performance Modular Offline Pipeline:** To eliminate ROS 2 message-copying/serialization overheads and guarantee 0% frame drops during dataset processing, we built a modular offline Python pipeline. It supports step-by-step debug execution (writing outputs to dedicated folders) and an end-to-end in-memory coordinator runner (writing only the final `thinning_plan.json`). (See [RUN_OFFLINE_PIPELINE.md](RUN_OFFLINE_PIPELINE.md)).
- **Large topics need RELIABLE QoS (2026-06-18).** The big messages (`sync_pair` ~9 MB, `masks`
  ~64 MB) were silently dropped by BEST_EFFORT subscribers, starving disparity and depth_fusion.
  Flipped the relevant subs/pubs to RELIABLE → pipeline runs through depth_fusion at 0.3 Hz.
  Also: close Isaac Sim (GPU starve), split GPUs, source venv per terminal. (See BRINGUP §11.)
- **Left/right were swapped** in the first run (fixed: left=`/rgb_static_1`).
- **IMU heading is unusable** — its yaw mis-tracks the U-turn (no magnetometer). So robot
  **heading comes from GNSS course-over-ground**, position from RTK. (See BRINGUP §9a/9b.)
- **Timestamps align** (camera JPG epoch ↔ bag ROS time) so pose interpolates per frame.
- Tree map: geojson has **238 tree points**, converted to the same local ENU frame.

## 6. The current blocker: calibration (get from supervisor)

World-frame flowers previously landed **~5–8 m off the trees** — a systematic offset, not noise.
Stereo intrinsics/extrinsics are now wired, but camera-to-platform mounting is still a
placeholder. We deliberately do **NOT** fit these to the tree map (that would be overfitting).
Remaining calibration still must come from real measurements:

1. **Confirm stereo calibration provenance** for the exact cameras/lenses that recorded the bag.
2. **Camera→platform mounting** (confirmed by user that `left` is the main camera side; position, orientation need validation).
3. **GNSS/IMU/camera lever arms** if available.

Until then, world output is structurally plausible but not metrically located.

## 7. What's left in Phase 1

- Confirm calibration provenance + camera mounting (`left` side confirmed) → re-check flowers land on trees.
- Steps 5-8 (`multiview_fusion → bio_sanity → tree_assignment → thinning_decision`) are fully wired, tested, and run end-to-end on the complete dataset in the offline pipeline.
- (later) Locate the missing `oskar_pruning` (Phase 2) code.
