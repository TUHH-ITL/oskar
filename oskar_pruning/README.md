# oskar_pruning

Real-time flower localization for the pruning phase, brought up against the
Isaac Sim two-by-two orchard.

The robot drives to a tree it has been told to prune, sweeps the wrist camera
over it, and publishes every flower it found with the tree it belongs to. What
to remove, and the approach and gripper that remove it, are later work packages.

## Setup

Prepare the scene once. This mounts the camera on the wrist, adds ground-truth
TF and the camera streams, and vendors the ZED X body so the scene no longer
depends on NVIDIA's S3 bucket. It writes a sibling file and leaves
`robot_and_orchard.usd` untouched.

```bash
cd oskar_simulation/isaac_sim
~/isaacsim_6_0_1/kit/python/bin/python3 ../scripts/patch_scene_for_pruning.py \
    robot_and_orchard.usd --output robot_and_orchard_pruning.usd
```

Regenerate the derived files whenever the scene's geometry changes. Both are
scene facts, not tuned values, so they are generated and never hand-edited:

```bash
python3 ../scripts/export_orchard_gt.py robot_and_orchard_pruning.usd ../../oskar_mapping/config
python3 ../scripts/export_camera_frames.py robot_and_orchard_pruning.usd ../../oskar_pruning/config/camera_frames_sim.yaml
```

Then open `robot_and_orchard_pruning.usd` in Isaac Sim, press Play, and:

```bash
ros2 launch oskar_pruning pruning_sim.launch.py
```

`mission:=false` brings up perception and mapping without driving, so a scan can
be requested by hand on `/pruning/scan_request`.

## Topics

| Topic | Type | Meaning |
|---|---|---|
| `/orchard/flower_map` | `oskar_msgs/PerTreeFlowers` | every flower found, per tree |
| `/orchard/flower_markers` | `visualization_msgs/MarkerArray` | the same, for RViz |
| `/pruning/detections_3d` | `oskar_msgs/FlowerDetections3D` | per-frame detections in the map frame |
| `/pruning/scan_request` | `oskar_msgs/ScanRequest` | scan one tree |
| `/pruning/scan_event` | `std_msgs/String` | scan lifecycle |
| `/pruning/status` | `oskar_msgs/PruningStatus` | mission progress |

`/pruning/reset_map` (`std_srvs/Trigger`) clears the map.

## Why this is not the Phase 1 chain

Phase 1 spreads perception over five nodes, and `FlowerMasks` carries a
full-frame `sensor_msgs/Image` per detected flower. A hundred flowers at
1280x720 is about 90 MB per frame on the wire, which no model speed recovers.
Here segmentation, depth and back-projection share one process, masks stay in
numpy, and only 3D points are published.

Fusion also differs. Phase 1 clusters with DBSCAN, which chains neighbouring
blossoms; the offline pipeline replaced it with frame-to-frame tracking. That
alone is still not enough for scanning, because a track dies when the flower
leaves view, so a sweep up and back counts every flower twice. Retired tracks
are therefore merged into a persistent landmark map by position. On a simulated
sweep of 40 flowers, up and back, that is 38 landmarks with the merge and 93
without.

## Bringing perception up

Three rungs, each isolating one thing. Only the mask source changes; everything
after it is the same code.

| Rung | Masks from | Isaac | Isolates |
|---|---|---|---|
| 1 | replayed ground truth | no | fusion, tree assignment, scoring |
| 2 | the renderer's instance segmentation (`detector: oracle`) | yes | deprojection, TF, sync, scan motion |
| 3 | Mask R-CNN (`detector: maskrcnn`) | yes | the detector alone |

Rung 1 needs no simulator and no GPU: run the stack with `mission:=false` and
publish `FlowerDetections3D` from `orchard_gt_sim.json`. It is what caught the
match radius being wrong, which would have looked like bad perception had the
detector been in the loop.

Rung 2 is why `patch_scene_for_pruning.py` labels the flower prims. Each flower
becomes one instance, so `/wrist_cam/left/instance_segmentation` carries perfect
masks on the real rendered image, and `/wrist_cam/left/instance_labels` maps
each id back to the prim it came from. Whatever recall this rung reaches is the
ceiling the detector is then measured against; a shortfall here is geometry or
timing, never the model.

Rung 3 changes one parameter. The gap to rung 2 is the detector's cost, and the
domain gap is the thing to watch: the scene renders synthetic flowers, so
`exp1_syn_syn` is the starting model and `exp3_real_real` is the field one.

## Parameters that need care

`match_radius_m` and `merge_radius_m` in `config/pruning_sim.yaml` are bounded
from both sides and must not be copied from the drive-by pipeline. Flowers in
this scene sit a median 30 mm apart, so the 0.06 m that suited a camera
advancing 0.2 m per frame fuses neighbours here: it cost 68% of the recall.
`merge_radius_m` wants roughly 2.5 times the position noise. Below that one
flower fragments into several landmarks, above it neighbours fuse. The two
bounds meet near 15 mm of noise; past that only corymbs are resolvable, not
individual flowers.

## Reach

The camera reaches 0.70 m to 3.95 m above the ground. That covers AppleTree_00,
_02 and _03 completely. AppleTree_01 grows to 4.31 m, so its top 0.36 m is out
of reach and `gt_eval_node` scores only the reachable band.
