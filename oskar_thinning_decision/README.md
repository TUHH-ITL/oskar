# oskar_thinning_decision

Which candidates stay on a tree. At most `max_kept` stay, as far apart as
possible: farthest-point insertion, then swaps that widen the closest kept pair.
Candidates the arm cannot reach, or failed to remove, are passed as `keep_ids`
and the rest is chosen around them.

| Interface | Type |
|---|---|
| `/decide_thinning` | `oskar_msgs/srv/DecideThinning` |

`selection.py` has no ROS dependency, so the same rule runs on the offline map.
