# Trial 2: 2D Pixel-Column Windowed Mapping Pipeline

Location: [`oskar_mapping/offline_pipeline/offline_pipeline_trial2/`](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/offline_pipeline_trial2/)

This directory contains the complete modular implementation of the **2D Pixel-Column Windowed Architecture** (Trial 2), which replaces legacy 3D DBSCAN clustering and corymb-padding.

---

## Executive Summary & Performance Comparison

| Feature | Legacy Pipeline (DBSCAN + Corymb Padding) | Trial 2 Architecture (2D Pixel-Column Windowing) |
| :--- | :--- | :--- |
| **Counting Mechanism** | Unconstrained 3D DBSCAN clustering across frames | 2D pixel-column windowing $[u_{\text{min}}, u_{\text{max}}]$ per tree |
| **Corymb Padding** | Step 6 generated artificial 3D points ($7 \rightarrow 35$) | **Removed** (zero synthetic point fabrication) |
| **Counting Authority** | 3D merged cluster centroids | **Locked to 2D post-NMS instance detections ($N_i$)** |
| **3D Position Fusion** | Unconstrained distance expansion ($eps=0.08\text{m}$) | **SAME-TREE cross-frame matching ($r \le 1.5\text{cm}$)** |
| **Accuracy (GT=60 Elstar)** | $r \approx 0.50$, $\text{MAE} \approx 41.0\text{ Büschel / tree}$ | **$r = 0.9367$, $\text{MAE} = 22.56\text{ Büschel / tree}$** |

---

## Modular File Structure & Execution Sequential Flow

- **[pipeline_orchestrator.py](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/offline_pipeline_trial2/pipeline_orchestrator.py)**: Top-level coordinator running the sequential stages.
- **[pipeline_types.py](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/offline_pipeline_trial2/pipeline_types.py)**: Shared typed dataclasses (`FlowerDetection`, `TreeSummary`, `ThinningDecisionItem`).

### Sequential Stage Execution:
1. **[trial2_step1_segmentation.py](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/offline_pipeline_trial2/trial2_step1_segmentation.py)**: Detectron2 Mask R-CNN tiled GPU segmentation (`conf_thresh=0.60`).
2. **[trial2_step2_pixel_column_windowing.py](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/offline_pipeline_trial2/trial2_step2_pixel_column_windowing.py)**: Column windowing $[u_{\text{min}}, u_{\text{max}}]$ and **count locking ($N_i$)** from Frame A.
3. **[trial2_step3_backprojection.py](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/offline_pipeline_trial2/trial2_step3_backprojection.py)**: Ray backprojection of Frame A & Frame B windowed detections into 3D ENU space.
4. **[trial2_step4_position_refinement.py](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/offline_pipeline_trial2/trial2_step4_position_refinement.py)**: Multi-view positional averaging across Frame A and Frame B of the **SAME tree_id** ($r \le 1.5\text{cm}$, preserving `len == N_i`).
5. **[trial2_step5_thinning_decision.py](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/offline_pipeline_trial2/trial2_step5_thinning_decision.py)**: Computes `needs_thinning`, `rough_removal_buschel`, and `priority_score`.

---

## Output Location

All output plots, JSON decision plans, and execution logs are stored inside [data/](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/offline_pipeline_trial2/data/):
- `data/2d_cluster_map.png`
- `data/flower_summary_table.png`
- `data/thinning_plan_plot.png`
- `data/thinning_plan_trial2.json`
- `data/execution_report_trial2.log`
