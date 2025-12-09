# Action message representing a grasp action with mask
# Copyright BSD-3-Clause

# Mask dimensions
uint32 width
uint32 height

# Mask data
uint8[] mask_data

# Action type (e.g., "parallel_jaw", "suction")
string action_type

# Action parameters (grasp-specific data)
float32[] action_data

# Confidence/quality score
float32 confidence
