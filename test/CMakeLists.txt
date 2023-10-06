#include test utils
include_directories(include)

if (ament_cmake_FOUND)
  add_subdirectory(ros2)
endif()
add_subdirectory(mppi-generic)
