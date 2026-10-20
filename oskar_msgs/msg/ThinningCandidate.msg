# One object the thinning can act on, built from a Landmark.
# pose.header.frame_id names the frame; the map frame in the pruning phase.

uint32 id                        # Landmark.landmark_id
uint32 tree_id
string class_label

geometry_msgs/PoseStamped pose   # position: [m] object centre
                                 # orientation: local +Z = approach axis, pointing out of the
                                 #   canopy into free space. Estimated as the mean direction the
                                 #   camera saw the object from, not a surface normal.
                                 #   Roll about +Z carries no information.

float32 diameter                 # [m], NaN = not estimated
float32 confidence               # [0..1] mean detector confidence
uint32 observation_count
