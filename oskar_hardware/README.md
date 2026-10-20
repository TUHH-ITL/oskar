# oskar_hardware

Hardware skeleton for the OSKAR blossom thinning robot. No real hardware
required -- every robot-facing call is faked, but the ROS 2 interfaces are
the real ones from `oskar_msgs`.

## Nodes

| Node | Endpoint | Type | Behaviour |
| --- | --- | --- | --- |
| `tool_node` | `/set_tool_state` | `oskar_msgs/srv/SetToolState` | State machine STOP / PREPARE / REMOVE, simulated durations |
| `compute_approach_pose_node` | `/compute_approach_pose` | `oskar_msgs/srv/ComputeApproachPose` | Pure geometry: blossom pose offset by `standoff` |
| `execute_approach_node` | `/execute_approach` | `oskar_msgs/action/ExecuteApproach` | TF transform, MoveIt planning, stand-off motion, and Cartesian approach |

`compute_approach_pose_node` has no side effects and should later move into a
planning package. `execute_approach_node` owns all robot mechanics and requires
MoveIt's `/move_action`, `/compute_cartesian_path`, and `/execute_trajectory`.

## Build

    cd ~/oskar_ws
    colcon build --symlink-install --packages-select oskar_hardware
    source install/setup.bash

## Run

    ros2 launch oskar_hardware oskar_hardware.launch.py

## Test in a second terminal

Tool, correct order:

    ros2 service call /set_tool_state oskar_msgs/srv/SetToolState "{command: 1}"
    ros2 service call /set_tool_state oskar_msgs/srv/SetToolState "{command: 2}"

Tool, error path (REMOVE without PREPARE) -- must report success: false:

    ros2 service call /set_tool_state oskar_msgs/srv/SetToolState "{command: 0}"
    ros2 service call /set_tool_state oskar_msgs/srv/SetToolState "{command: 2}"

Compute from an identity blossom orientation. The result position must be
z + standoff, with tool +Z oriented back toward the blossom:

    ros2 service call /compute_approach_pose oskar_msgs/srv/ComputeApproachPose "{blossom: {id: 1, pose: {header: {frame_id: tree_1}, pose: {position: {x: 0.5, y: 0.1, z: 1.2}}}, confidence: 0.9}, standoff: 0.15}"

Expected: `success: true`, position `(0.5, 0.1, 1.35)` in frame `tree_1`, and
orientation `(x=1, y=0, z=0, w=0)` so a positive execution distance moves
toward the blossom.

Execute a stand-off pose, then move 0.10 m along the tool's local +Z axis:

    ros2 action send_goal --feedback /execute_approach oskar_msgs/action/ExecuteApproach "{target_pose: {header: {frame_id: base_link}, pose: {position: {x: 0.5, y: 0.0, z: 0.5}, orientation: {w: 1.0}}}, approach_distance: 0.10}"

The execution node parameters default to planning group `ur_manipulator`, tool
link `tool0`, and planning frame `base_link`. Only one execution goal is
accepted at a time.

## ExecuteApproach errors

The result uses a small, stable OSKAR taxonomy and also returns
`moveit_error_code`, the original `moveit_msgs/MoveItErrorCodes` value when
MoveIt produced the failure.

| Category | Meaning | Typical response |
| --- | --- | --- |
| `ERROR_INVALID_GOAL` | Missing frame, invalid quaternion, or invalid distance | Fix the caller |
| `ERROR_TF` | Target frame cannot be transformed into the planning frame | Check TF and timestamps |
| `ERROR_MOVEIT_UNAVAILABLE` | A required MoveIt action/service is absent | Start or repair MoveIt |
| `ERROR_NO_IK` | No inverse-kinematics solution exists | Change pose or tool orientation |
| `ERROR_COLLISION` | Start or goal state is in collision | Update scene or choose another pose |
| `ERROR_PLAN_FAILED` | A valid request could not produce a motion plan | Retry or change constraints |
| `ERROR_CARTESIAN_INCOMPLETE` | Linear approach fraction is below the configured threshold | Shorten or reposition the approach |
| `ERROR_CONTROLLER` | Controller rejected or failed trajectory execution | Check ros2_control and robot mode |
| `ERROR_EXECUTION_ABORTED` | Other execution failure or MoveIt rejection | Inspect message and native code |
| `ERROR_CANCELLED` | Caller cancellation or MoveIt preemption | Expected terminal state |
| `ERROR_TIMEOUT` | Server, planning, or execution deadline expired | Check load and connectivity |

This keeps callers independent of MoveIt enum changes while preserving the
native code for logs and diagnosis.

## Next steps

* Replace `_hw_stop` / `_hw_prepare` / `_hw_remove` in `tool_node.py` with
  real driver calls; keep the service contract identical.
* Move `compute_approach_pose_node` into the planning package.
* Add the action servers (`ScanRegion`, `PlanRemovalSequence`,
  `RemoveTargets`) on top of these services.
