# Which candidates stay on one tree. Written by DecideThinning; deciding again
# makes a new one. No poses and no order: PlanRemovalSequence adds those.

std_msgs/Header header
uint32 tree_id
uint32 max_kept                  # upper limit the decision was made for
uint32[] keep_ids
uint32[] remove_ids
float32 min_spacing              # [m] closest pair among the kept ones, NaN if fewer than two
bool limit_met                   # false when more must stay than the limit allows
string message
