# OSKAR: Apple Flower Mapping and Thinning Decision System

A complete ROS 2 Humble implementation for a pruning robot that autonomously maps apple flowers in an orchard and makes thinning decisions using stereo vision, deep learning, and biology-informed inference.

## Quick Start

### Build
```bash
cd ~/ros2_ws
bash build_oskar.sh
```

### Launch (Isaac Sim - Phase 1)
```bash
source install/setup.bash
ros2 launch oskar_mapping mapping_sim.launch.py
```

### Launch (Real Robot - Phase 1)
```bash
source install/setup.bash
ros2 launch oskar_mapping mapping.launch.py
```

### Launch (Phase 2 Pruning)
```bash
# Simulation
ros2 launch oskar_pruning pruning_sim.launch.py

# Real robot
ros2 launch oskar_pruning pruning.launch.py
```

## Project Overview

The OSKAR system operates in two phases:

### **Phase 1: Flower Mapping** (during tree traversal)
1. Stereo synchronization and rectification
2. Disparity estimation (Fast-FoundationStereo)
3. Apple flower detection (Detectron2 Mask R-CNN)
4. Depth fusion and 3D backprojection
5. Multi-view landmark fusion (DBSCAN clustering)
6. Apple biology sanity checking (corymb inference)
7. Per-tree flower assignment
8. Thinning decision (coarse count-based)

**Outputs**: Per-tree flower counts (confirmed + inferred), thinning plan (need to thin? how many to remove?)

### **Phase 2: Pruning Execution** (during arm deployment)
1. Local refinement with end-effector camera (close-range re-detection)
2. Flower ranking by agronomic priority
3. Closed-loop pruning execution (target-count driven)

**Outputs**: Pruning target poses, execution status

## Repository Structure

```
oskar_perception/
├── CLAUDE.md                    # Original specification document
├── IMPLEMENTATION_SUMMARY.md    # Implementation details and design decisions
├── README.md                    # This file
├── oskar_msgs/                  # Custom message type package
│   ├── msg/                     # 11 message type definitions
│   ├── CMakeLists.txt
│   └── package.xml
├── oskar_mapping/               # Phase 1 mapping pipeline (9 ROS nodes)
│   ├── oskar_mapping/
│   │   ├── stereo_sync_node.py           # Rectification
│   │   ├── disparity_node.py             # Fast-FoundationStereo wrapper
│   │   ├── segmentation_node.py          # Detectron2 flower detection
│   │   ├── depth_fusion_node.py          # Mask erosion + median depth
│   │   ├── backprojection_node.py        # TF2 transform to map frame
│   │   ├── multiview_fusion_node.py      # DBSCAN clustering + rolling buffer
│   │   ├── bio_sanity_node.py            # Corymb inference
│   │   ├── tree_assignment_node.py       # Tree position matching
│   │   └── thinning_decision_node.py     # Per-tree thinning decisions
│   ├── launch/
│   │   ├── mapping_sim.launch.py         # Isaac Sim launch
│   │   └── mapping.launch.py             # Real robot launch
│   ├── config/
│   │   ├── mapping_params_sim.yaml       # Simulation parameters
│   │   ├── mapping_params.yaml           # Real robot parameters
│   │   ├── tree_map_sim.yaml             # Simulation tree positions
│   │   └── tree_map.yaml                 # Real robot orchard layout
│   ├── tests/                   # Unit tests (not yet implemented)
│   ├── setup.py
│   └── package.xml
└── oskar_pruning/               # Phase 2 pruning pipeline (2 ROS nodes)
    ├── oskar_pruning/
    │   ├── local_refinement_node.py      # Close-range flower refinement
    │   └── pruning_decision_node.py      # Ranking + closed-loop control
    ├── launch/
    │   ├── pruning_sim.launch.py         # Isaac Sim launch
    │   └── pruning.launch.py             # Real robot launch
    ├── config/
    │   ├── pruning_params_sim.yaml       # Simulation parameters
    │   └── pruning_params.yaml           # Real robot parameters
    ├── setup.py
    └── package.xml
```

## ROS Topic Graph

### Phase 1 (Mapping)

```
Isaac Sim / Real Robot
├── /rgb_camera_1 (or /camera/left/image_raw)
├── /rgb_camera_2 (or /camera/right/image_raw)
├── /rgb_camera_1/camera_info (or /camera/left/camera_info)
└── /rgb_camera_2/camera_info (or /camera/right/camera_info)
    ↓ [stereo_sync_node]
    /stereo/sync_pair (StereoPair)
    ├── [disparity_node] → /stereo/disparity, /stereo/depth
    └── [segmentation_node] → /flowers/masks (FlowerMasks)
        ↓ [depth_fusion_node]
        /flowers/detections_3d (FlowerDetections3D)
        ↓ [backprojection_node]
        /flowers/world_obs (FlowerLandmarks)
        ↓ [multiview_fusion_node]
        /flowers/landmarks (FlowerLandmarks)
        ↓ [bio_sanity_node]
        /flowers/landmarks_bio (FlowerLandmarks)
        ↓ [tree_assignment_node]
        /orchard/per_tree_flowers (PerTreeFlowers)
        ↓ [thinning_decision_node]
        /orchard/thinning_plan (ThinningPlan)
```

### Phase 2 (Pruning)

```
/orchard/thinning_plan
├── [local_refinement_node]
│   └── Requires: /ee_camera/image_color, /ee_camera/depth, /ee_camera/camera_info
│       Publishes: /pruning/targets (PruningTargets)
│           ↓ [pruning_decision_node]
│           /pruning/next_target (PoseStamped)
│           /pruning/status (PruningStatus)
│           Subscribes: /pruning/cut_complete (Bool) — feedback from hardware
```

## Key Features

✅ **Fully Parameterized**: All environment-specific differences (topic names, frame IDs, scales, thresholds) are ROS parameters. Single codebase works on Isaac Sim and real robot without modification.

✅ **Simulation-First Development**: All initial implementation happens in Isaac Sim with identical code and config (just different parameter files).

✅ **Biology-Informed Inference**: Apple corymbs (clusters of ~5 flowers) are inferred from partial detections, accounting for occlusion.

✅ **Multi-View Fusion**: Observations across multiple frames are clustered (DBSCAN) and confidence-weighted.

✅ **Robust Error Handling**: TF lookups don't crash, old observations are flushed by both robot position and time-based fallback.

✅ **Closed-Loop Pruning**: The pruning node counts removed flowers and stops when target is reached.

## Isaac Sim Prerequisites

Before running Phase 1 mapping on Isaac Sim, verify:

- [ ] `/rgb_camera_1` and `/rgb_camera_2` publish at ~5 Hz  
  `ros2 topic hz /rgb_camera_1`
- [ ] `/rgb_camera_1/camera_info` and `/rgb_camera_2/camera_info` contain valid K and P matrices  
  `ros2 topic echo /rgb_camera_1/camera_info`
- [ ] Stereo baseline reads correctly from CameraInfo  
  `P[3] / -P[0]` should be ~0.10–0.20 m
- [ ] TF tree connects cameras to base_link  
  `ros2 run tf2_tools view_frames`
- [ ] `/clock` is published for ROS simulation time  
  `ros2 topic hz /clock`

See CLAUDE.md for full Isaac Sim configuration checklist.

## Real Robot Prerequisites

Before deploying to the real robot:

- [ ] Stereo cameras streaming at /camera/left/image_raw and /camera/right/image_raw
- [ ] CameraInfo topics with correct intrinsics
- [ ] TF tree: map → odom → base_link connected
- [ ] End-effector depth camera on /ee_camera/* (Phase 2)
- [ ] Pruning arm control listening to /pruning/next_target poses

Switch launch file and parameter file (see Quick Start above).

## Dependencies

### ROS 2 Packages
```
rclpy, sensor_msgs, geometry_msgs, stereo_msgs, std_msgs
tf2_ros, tf2_geometry_msgs, message_filters, cv_bridge
```

### Python Packages
```
torch>=2.6.0 (CUDA 12.4)
torchvision>=0.21.0
detectron2        # Install from: https://github.com/facebookresearch/detectron2
scikit-learn      # for DBSCAN
opencv-python, numpy, pyyaml, scipy
```

### External Models/Data
```
Fast-FoundationStereo checkpoint (23-36-37):
  Clone: https://github.com/NVlabs/Fast-FoundationStereo
  Download: weights/23-36-37/
  
Detectron2 Mask R-CNN (Apple flowers):
  Path: /home/workstation/oskar/2025-transformers-for-apple-flower-segmentation-bhangale/...
  Model: model_final.pth
  Config: config.yaml (must be in same directory)
```

## Configuration

### Simulation (Isaac Sim)
- **Parameter file**: `config/mapping_params_sim.yaml`
- **use_sim_time**: true
- **Camera topics**: /rgb_camera_1, /rgb_camera_2
- **Camera frames**: static_camera_1, static_camera_2
- **Disparity scale**: 1.0 (720p, no downsampling)
- **Segmentation threshold**: 0.3 (lower for sim domain gap)
- **Erosion radius**: 3 pixels

### Real Robot
- **Parameter file**: `config/mapping_params.yaml`
- **use_sim_time**: false
- **Camera topics**: /camera/left/image_raw, /camera/right/image_raw
- **Camera frames**: camera_left_optical, camera_right_optical
- **Disparity scale**: 0.5 (20 MP → ~2.7 MP, then upsample)
- **Segmentation threshold**: 0.5 (tuned on real images)
- **Erosion radius**: 5 pixels

### Tree Map
Edit `config/tree_map_sim.yaml` or `config/tree_map.yaml` to match your orchard layout:
```yaml
trees:
  1: {x: 0.0, y: 0.0}
  2: {x: 1.0, y: 0.0}
  3: {x: 2.0, y: 0.0}
```

## Visualization

Use RViz to visualize landmarks and thinning plans:

```bash
ros2 run rviz2 rviz2
```

Add markers for:
- `/flowers/landmarks_bio` (confirmed + inferred flowers)
- `/orchard/per_tree_flowers` (per-tree grouping)
- Robot pose in `map` frame
- Camera frustums (static_camera_1, static_camera_2 for sim)

## Rosbag Recording

Useful topics to record:
```bash
ros2 bag record \
  /stereo/sync_pair \
  /flowers/masks \
  /stereo/depth \
  /flowers/detections_3d \
  /flowers/world_obs \
  /flowers/landmarks \
  /flowers/landmarks_bio \
  /orchard/per_tree_flowers \
  /orchard/thinning_plan \
  /tf /tf_static
```

Replay:
```bash
ros2 bag play oskar_mapping_sim
```

## Implementation Notes

### Simulation-to-Real Transfer

All parameters that differ between Isaac Sim and the real robot are externalized:

| Parameter | Sim | Real Robot |
|-----------|-----|-----------|
| use_sim_time | true | false |
| Left image topic | /rgb_camera_1 | /camera/left/image_raw |
| Right image topic | /rgb_camera_2 | /camera/right/image_raw |
| Left camera frame | static_camera_1 | camera_left_optical |
| Right camera frame | static_camera_2 | camera_right_optical |
| Disparity scale | 1.0 | 0.5 |
| Confidence threshold | 0.3 | 0.5 |
| Erosion radius | 3 px | 5 px |

No node source code changes needed — just swap the parameter file.

### Known Limitations

1. **Fast-FoundationStereo**: Placeholder model class. Full integration requires actual FFM wheel + checkpoint.
2. **Detectron2 Config**: Expected in model directory. Update path if moved.
3. **Unit Tests**: Not yet implemented. Would test node logic with saved ROS messages.

## Future Work

- [ ] Write comprehensive unit tests for all nodes
- [ ] Validate on Isaac Sim with sample camera feeds
- [ ] Rosbag playback and quantitative evaluation
- [ ] Real robot field trials
- [ ] Performance profiling (FPS, latency, memory)
- [ ] RViz plugin for real-time monitoring
- [ ] Agronomic validation (correlation with actual fruit thinning needs)

## References

- **OSKAR Specification**: See `CLAUDE.md` for complete system requirements
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md` for design decisions
- **Apple Biology**: Corymbs typically have 5 flowers (~2–4 cm spacing)

## License

Apache 2.0

## Authors

OSKAR Development Team
April 2026
