# One object fused over many frames into a persistent map entry.
uint32 landmark_id
string class_label
geometry_msgs/Point position
uint32 observation_count
float32 mean_confidence
bool is_inferred
uint32 tree_id
# Mean of the unit vectors toward the camera over all observations; zero if unknown.
geometry_msgs/Vector3 view_direction
# [m] median over observations, 0 = not estimated
float32 diameter
