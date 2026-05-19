cmake_minimum_required(VERSION 3.8)
project(ur10e_robotiq_2f140_combined)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    launch
    meshes
    urdf
  DESTINATION
    share/${PROJECT_NAME}
)

ament_package()
