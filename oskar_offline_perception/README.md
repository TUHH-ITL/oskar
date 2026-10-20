# oskar_offline_perception

Mapping phase: flowers per tree from the Basler stereo recording of a row.
Plain Python, no ROS; `COLCON_IGNORE` keeps colcon out of the data directories.

```bash
python3 run_pipeline.py --max-frames -1
```

| Step | Module |
|---|---|
| 1 | `step1_segmentation(_tiled).py`: instance masks, through `oskar_detection` |
| 2–3 | `step2_disparity.py`, `step3_depth_fusion.py`: depth per flower |
| 4 | `step4_backprojection.py`: 3D in the map frame |
| 5 | `step5_*_fusion.py`: one landmark per flower across frames |
| 6–8 | bio sanity, tree assignment, thinning decision |

Paths and settings are in `config.py`; the model is `SEGMENTATION_MODEL`, a
model file name from `oskar_detection/models`. Tree positions come from
`oskar_orchard/config/tree_map_real.yaml`. Outputs go to `data/`.
