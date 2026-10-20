# oskar_orchard

Orchard facts every other package reads. Nothing here is tuned.

| File | Content |
|---|---|
| `config/tree_map_real.yaml` | surveyed trunk positions, Elstar row A27 |
| `config/tree_map_sim.yaml` | trunk positions of the Isaac Sim orchard |
| `config/orchard_gt_sim.json` | every flower of the Isaac Sim orchard, from `oskar_simulation/scripts/export_orchard_gt.py` |

ROS nodes find the files with `get_package_share_directory("oskar_orchard")`;
the offline perception reads them from the source tree.
