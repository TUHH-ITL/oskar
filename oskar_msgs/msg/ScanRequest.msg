std_msgs/Header header
uint32 tree_id
# Distance the camera keeps from the trunk axis, in metres.
float32 standoff_m
# Height band to sweep, in the map frame.
float32 z_min
float32 z_max
# Vertical passes, offset along the row; 1 is a single sweep up the near face.
uint32 columns
