cmake_minimum_required(VERSION 3.8)
project(tipard_ur20_combined)

find_package(ament_cmake REQUIRED)

install(
  FILES tipard_ur20_combined.urdf
  DESTINATION share/${PROJECT_NAME}/urdf
)

install(
  DIRECTORY meshes
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
