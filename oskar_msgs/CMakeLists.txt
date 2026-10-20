cmake_minimum_required(VERSION 3.8)
project(oskar_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(msg_files
  # perception
  "msg/Detection3D.msg"
  "msg/Detections3D.msg"
  "msg/Landmark.msg"
  "msg/Landmarks.msg"
  "msg/PerTreeLandmarks.msg"
  "msg/ScanRequest.msg"
  # mission
  "msg/MissionStatus.msg"
  # thinning decision
  "msg/ThinningCandidate.msg"
  "msg/ThinningCandidates.msg"
  "msg/ThinningDecision.msg"
  # removal
  "msg/RemovalTarget.msg"
  "msg/RemovalSequence.msg"
  "msg/RemovalResult.msg"
)

set(srv_files
  "srv/DecideThinning.srv"
  "srv/ComputeApproachPose.srv"
  "srv/SetToolState.srv"
)

set(action_files
  "action/ExecuteApproach.action"
  "action/PlanRemovalSequence.action"
  "action/RemoveTargets.action"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  ${action_files}
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs
)

ament_package()
