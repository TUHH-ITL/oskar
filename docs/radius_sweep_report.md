# Apple Flower Mapping: Canopy Assignment Radius Sweep Report

This report presents a post-hoc sweep analysis of the flower-to-tree assignment radius constraint (`max_radius`) performed on the full 2,566-frame dataset (containing `14,875` landmarks).

---

## 1. Radius Sweep Results Table

| Radius (m) | Overlap Count | Overlap % | Orphan Count | Orphan % | Trees Scanned (>0) | Zero-Flower Trees | Total Assigned Flowers |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.20** | 0 | 0.00% | 11,085 | 74.52% | 174 / 238 | 64 | 3,790 |
| **0.25** | 0 | 0.00% | 9,609 | 64.60% | 187 / 238 | 51 | 5,266 |
| **0.30** | 9 | 0.06% | 7,919 | 53.24% | 193 / 238 | 45 | 6,956 |
| **0.35** | 30 | 0.20% | 6,211 | 41.75% | 199 / 238 | 39 | 8,664 |
| **0.40** | 165 | 1.11% | 4,737 | 31.85% | 201 / 238 | 37 | 10,138 |
| **0.45** | 493 | 3.31% | 3,310 | 22.25% | 206 / 238 | 32 | 11,565 |
| **0.50** | 1,089 | 7.32% | 2,124 | 14.28% | 209 / 238 | 29 | 12,751 |
| **0.60** | 3,379 | 22.72% | 1,321 | 8.88% | 210 / 238 | 28 | 13,554 |
| **0.70** | 6,369 | 42.82% | 1,028 | 6.91% | 210 / 238 | 28 | 13,847 |
| **0.80** | 9,224 | 62.01% | 818 | 5.50% | 210 / 238 | 28 | 14,057 |

*Note: The raw sweep metrics are saved in [radius_sweep_results.json](file:///home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/data/data_archive/radius_sweep_2026-07-02/radius_sweep_results.json).*

---

## 2. Crossover Analysis & Sweet Spot

* **The Pathology (0.8m)**: At the current configuration radius of `0.8m`, the overlap rate is a massive **62.01%** (9,224 landmarks). Because the average tree-to-tree spacing is only **~0.98m**, neighboring assignment circles overlap across almost the entire row. This makes flower-to-tree assignments highly sensitive to minor camera calibration offsets or GPS/odometry drift.
* **The Sweet Spot (0.50m)**: The range of **0.45m to 0.55m** (ideally **0.50m**) represents the optimal candidate crossover:
  * **Overlap Reduction**: At `0.50m`, the overlap rate drops from **62.01% to 7.32%**, virtually resolving the assignment ambiguity.
  * **Coverage Retention**: The orphan rate is kept to a moderate **14.28%** (meaning `85.72%` of all detected flower landmarks are successfully assigned to trees).
  * **Tree Scan Stability**: The scanned tree counts remain highly stable: **209 / 238 trees scanned** (losing only 1 tree compared to the baseline of 210) and **29 zero-flower trees** (compared to 28 at the baseline). 
  * **Conclusion**: Shifting the assignment radius to `0.50m` fixes the overlap pathology without losing tree coverage.
