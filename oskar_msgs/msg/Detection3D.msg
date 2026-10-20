# One detected object (flower, fruitlet, ...) placed in 3D, from one frame.
uint32 instance_id
string class_label               # the detector's class, e.g. "flower"
geometry_msgs/Point position
float32 confidence
bool low_confidence
float32 depth_std_ratio
# Unit vector from the object toward the camera, in the header frame.
geometry_msgs/Vector3 view_direction
# [m] from mask area and depth, 0 = not estimated
float32 diameter
