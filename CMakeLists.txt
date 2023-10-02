cmake_minimum_required(VERSION 3.8)
project(mppi_paper_examplesi CXX CUDA)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/subtree/MPPI-Generic/cmake")
include(MPPIGenericToolsConfig)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(BUILD_TESTS OFF CACHE BOOL "Build MPPI-Generic Tests")
set(BUILD_EXAMPLES OFF CACHE BOOL "Build MPPI-Generic Examples")

# Other necessary packages
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
