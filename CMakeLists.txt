cmake_minimum_required(VERSION 3.8)
project(mppi_paper_examples CXX CUDA)

set(CMAKE_CXX_STANDARD 14) # use C++14 standard as gtest in Ubuntu 24.04 requires it

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/submodules/MPPI-Generic/cmake")
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
    find_package(${PKG} QUIET)
  endforeach()
endif()
find_package(OpenCV REQUIRED)

include_directories(SYSTEM
  ${OpenCV_INCLUDE_DIRS}
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/submodules/MPPI-Generic)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)

set(BUILD_TESTS ON)
if (BUILD_TESTS)
  if (ament_cmake_FOUND)
    find_package(ament_cmake_gtest REQUIRED)
  else()
    set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/submodules/MPPI-Generic/cmake/Modules")
    ############################################################
    # copied from
    # https://github.com/google/googletest/tree/master/googletest#incorporating-into-an-existing-cmake-project
    ############################################################
    # Download and unpack googletest at configure time
    if (NOT TARGET gtest_main)
      list(GET CMAKE_MODULE_PATH -1 MPPI_GENERIC_MODULES)
      configure_file(${MPPI_GENERIC_MODULES}/CMakeLists.txt.gtest.in
                     ${PROJECT_BINARY_DIR}/googletest-download/CMakeLists.txt)
      execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/googletest-download )
      if(result)
        message(FATAL_ERROR "CMake step for googletest failed: ${result}")
      endif()
      execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/googletest-download )
      if(result)
        message(FATAL_ERROR "Build step for googletest failed: ${result}")
      endif()

      # Prevent overriding the parent project's compiler/linker
      # settings on Windows
      set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

      # Prevent googletest installation
      set(INSTALL_GTEST OFF)

      # Add googletest directly to our build. This defines
      # the gtest and gtest_main targets.
      add_subdirectory(${PROJECT_BINARY_DIR}/googletest-src
                       ${PROJECT_BINARY_DIR}/googletest-build)
    endif()

    include(GoogleTest)
  endif()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif()

if (ament_cmake_FOUND)
  ament_package()
endif()
