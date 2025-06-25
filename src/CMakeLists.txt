file(GLOB CUDA_TARGET_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cu)

foreach(T_FILE IN LISTS CUDA_TARGET_SRCS)
  # Get filename without extension to use as target name
  get_filename_component(T_NAME ${T_FILE} NAME_WE)
  add_executable(${T_NAME} ${T_FILE})
  target_include_directories(${T_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_compile_options(${T_NAME} PUBLIC "--diag-suppress=20012")
  target_link_libraries(${T_NAME} PUBLIC MPPI::MPPI
                        )
  # set_target_properties(${T_NAME} PROPERTIES FOLDER test)
endforeach()

# Create ROS2 executable if possible
if (nav2_mppi_controller_FOUND)
  set(ROS2_EXECUTABLE ros2_mppi)
  add_executable(${ROS2_EXECUTABLE} ROS2_MPPI.cpp)
  # ament_target_dependencies(${ROS2_EXECUTABLE}
  #   rclcpp
  #   nav2_mppi_controller
  # )
  target_link_libraries(${ROS2_EXECUTABLE} PUBLIC
      nav2_mppi_controller::mppi_controller
      nav2_mppi_controller::mppi_critics
      rclcpp::rclcpp
    )

  # Nav2 MPPI API changed between ROS 2 iron and ROS 2 kilted so I need a flag to
  # detect the difference
  if (DEFINED ENV{ROS_DISTRO} AND "$ENV{ROS_DISTRO}" STREQUAL "iron")
    target_compile_definitions(${ROS2_EXECUTABLE} PUBLIC CMAKE_ROS_IRON)
  endif()

  install(TARGETS
    ${ROS2_EXECUTABLE}
    DESTINATION lib/${PROJECT_NAME}
  )
endif()
