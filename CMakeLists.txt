cmake_minimum_required(VERSION 3.8)
project(mppi_paper_examples CXX CUDA)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/subtree/MPPI-Generic/cmake")
set(CMAKE_CUDA_ARCHITECTURES "86")
include(MPPIGenericToolsConfig)
# set(CMAKE_CUDA_FLAGS "-arch=sm_86")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(BUILD_TESTS OFF CACHE BOOL "Build MPPI-Generic Tests")
set(BUILD_EXAMPLES OFF CACHE BOOL "Build MPPI-Generic Examples")

# Other necessary packages
find_package(ament_cmake REQUIRED)
set(ROS2_PKGS nav2_mppi_controller rclcpp)
foreach(PKG IN LISTS ROS2_PKGS)
  find_package(${PKG} REQUIRED)
endforeach()
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(SYSTEM
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${OpenCV_INCLUDE_DIRS}
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/subtree/MPPI-Generic/submodules/cnpy ${CMAKE_BINARY_DIR}/subtree/MPPI-Generic/submodules/cnpy)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/subtree/MPPI-Generic ${CMAKE_BINARY_DIR}/subtree/MPPI-Generic)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)

ament_package()
