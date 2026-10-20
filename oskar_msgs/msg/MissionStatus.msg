# Progress of the mission on the current tree.
std_msgs/Header header
uint32 tree_id
uint32 current_count             # objects mapped, then left after removal
uint32 target_count              # upper limit; 0 while only scanning
bool done
