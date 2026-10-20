# oskar_mission

The pruning phase end to end: drive to each tree from the offline perception's
list, scan it, and, with `thin:=true`, decide which objects stay and remove the
rest, in rounds until the limit holds.

```bash
ros2 launch oskar_mission mission_sim.launch.py            # scan only
ros2 launch oskar_mission mission_sim.launch.py thin:=true # scan and thin
```

Parameters are in `config/mission_sim.yaml`; `max_kept_per_tree` is the upper
limit of objects left on a tree.
