set(GTEST_LIBRARIES gtest gmock gtest_main)
file(GLOB TARGET_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

foreach(T_FILE IN LISTS TARGET_SRCS)
  get_filename_component(T_NAME ${T_FILE} NAME_WE)
  # Time out after 20 minutes
  ament_add_gtest(${T_NAME} ${PROJECT_SOURCE_DIR}/test/test_main.cpp ${T_FILE}
    TIMEOUT 1200)
  target_link_libraries(${T_NAME}
    ${GTEST_LIBRARIES}
  )

  target_include_directories(${T_NAME} SYSTEM PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

  # Nav2 MPPI API changed between ROS 2 iron and ROS 2 kilted so I need a flag to
  # detect the difference
  if (DEFINED ENV{ROS_DISTRO} AND "$ENV{ROS_DISTRO}" STREQUAL "iron")
    target_compile_definitions(${T_NAME} PUBLIC CMAKE_ROS_IRON)
  endif()

  # ament_target_dependencies(${T_NAME}
  #   rclcpp
  #   nav2_mppi_controller
  # )
  target_link_libraries(${T_NAME}
    nav2_mppi_controller::mppi_controller
    nav2_mppi_controller::mppi_critics
    rclcpp::rclcpp
  )
  # gtest_add_tests(TARGET ${T_NAME})
  set_target_properties(${T_NAME} PROPERTIES FOLDER test)
endforeach()
