# GQCNNGrasp message representing a planned grasp
# Copyright BSD-3-Clause

# Grasp pose in 3D space
geometry_msgs/Pose pose

# Predicted grasp quality (Q-value)
float64 q_value

# Grasp type constants
uint8 PARALLEL_JAW=0
uint8 SUCTION=1

# Grasp type (parallel jaw or suction)
uint8 grasp_type

# Grasp center in pixel coordinates [x, y]
float64[2] center_px

# Grasp angle in radians (for parallel jaw)
float64 angle

# Grasp depth
float64 depth

# Thumbnail image of the grasp region
sensor_msgs/Image thumbnail
