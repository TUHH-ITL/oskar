# Outcome of one removal attempt.

uint32 candidate_id
bool success

uint8 failure_reason
uint8 FAILURE_NONE=0
uint8 FAILURE_UNREACHABLE=1      # no valid IK solution
uint8 FAILURE_PLANNING=2         # motion planner found no path
uint8 FAILURE_COLLISION=3        # aborted due to collision / force limit
uint8 FAILURE_END_EFFECTOR=4     # end-effector fault or failed actuation
uint8 FAILURE_NOT_FOUND=5        # object no longer visible at expected pose
uint8 FAILURE_ABORTED=6          # cancelled by operator or higher level

builtin_interfaces/Duration duration
string message
