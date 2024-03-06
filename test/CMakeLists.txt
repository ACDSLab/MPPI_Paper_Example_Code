#include test utils
include_directories(include)

# Create header files pointing to autorally nn and track information
set(HEADER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/templated_headers/)

file(GLOB HEADERS ${HEADER_PATH}*)
foreach(T_FILE IN LISTS HEADERS)
    get_filename_component(T_NAME ${T_FILE} NAME_WE)
    string(REPLACE ".in" "" STRIPPED_NAME ${T_NAME})
    set(TEMPLATED_HEADER_FILE ${PROJECT_BINARY_DIR}/include/${STRIPPED_NAME}.h)
    configure_file(${T_FILE} ${TEMPLATED_HEADER_FILE})
    message(STATUS "Generated ${TEMPLATED_HEADER_FILE}")
endforeach()
include_directories(${PROJECT_BINARY_DIR}/include)

if (ament_cmake_FOUND)
  add_subdirectory(ros2)
endif()
add_subdirectory(mppi-generic)
add_subdirectory(autorally)
add_subdirectory(autorally_sys)
