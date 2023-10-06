cmake_minimum_required(VERSION 3.8)
project(mppi_paper_examples CXX CUDA)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/subtree/MPPI-Generic/cmake")
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.18.0")
  cmake_policy(SET CMP0104 OLD)
  # if (CUDA_VERSION VERSION_GREATER_EQUAL "11.8.0")
  #   message(WARNING "We can use native?")
  #   set(CMAKE_CUDA_ARCHITECTURES "native")
  # else()
  #   set(CMAKE_CUDA_ARCHITECTURES "52")
  # endif()
endif()
include(MPPIGenericToolsConfig)
# set(CMAKE_CUDA_FLAGS "-arch=sm_86")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(BUILD_TESTS OFF CACHE BOOL "Build MPPI-Generic Tests")
set(BUILD_EXAMPLES OFF CACHE BOOL "Build MPPI-Generic Examples")

# Other necessary packages
find_package(ament_cmake QUIET)
if (ament_cmake_FOUND)
  set(ROS2_PKGS nav2_mppi_controller rclcpp)
  foreach(PKG IN LISTS ROS2_PKGS)
    find_package(${PKG} REQUIRED)
  endforeach()
endif()
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(SYSTEM
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${OpenCV_INCLUDE_DIRS}
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
if (NOT BUILD_TESTS)
  # WIll be built as part of MPPI-Generic if BUILD_TESTS is true
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/subtree/MPPI-Generic/submodules/cnpy ${CMAKE_BINARY_DIR}/subtree/MPPI-Generic/submodules/cnpy)
endif()
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/subtree/MPPI-Generic ${CMAKE_BINARY_DIR}/subtree/MPPI-Generic)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)

set(BUILD_TESTS ON)
if (BUILD_TESTS)
  if (ament_cmake_FOUND)
    find_package(ament_cmake_gtest REQUIRED)
  endif()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif()

if (ament_cmake_FOUND)
  ament_package()
endif()
