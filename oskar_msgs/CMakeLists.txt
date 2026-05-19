cmake_minimum_required(VERSION 3.8)
project(oskar_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

set(msg_files
  "msg/StereoPair.msg"
  "msg/FlowerMask.msg"
  "msg/FlowerMasks.msg"
  "msg/FlowerDetection3D.msg"
  "msg/FlowerDetections3D.msg"
  "msg/FlowerLandmark.msg"
  "msg/FlowerLandmarks.msg"
  "msg/PerTreeFlowers.msg"
  "msg/ThinningPlan.msg"
  "msg/PruningTargets.msg"
  "msg/PruningStatus.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES geometry_msgs std_msgs sensor_msgs
)

ament_package()
