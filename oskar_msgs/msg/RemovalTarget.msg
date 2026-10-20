# One object to remove and how the tool gets there.

oskar_msgs/ThinningCandidate candidate
geometry_msgs/PoseStamped approach_pose   # tool tip at the stand-off, +Z pointing at the object
float32 approach_distance                 # [m] linear move along tool +Z from approach_pose
