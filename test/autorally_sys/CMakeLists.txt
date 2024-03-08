set(GTEST_LIBRARIES gtest gtest_main)
file(GLOB TARGET_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cu)

foreach(T_FILE IN LISTS TARGET_SRCS)
  get_filename_component(T_NAME ${T_FILE} NAME_WE)
  # Time out after 20 minutes
  # ament_add_gtest(${T_NAME} ${PROJECT_SOURCE_DIR}/test/test_main.cpp ${T_FILE}
  #   TIMEOUT 1200)
  # target_link_libraries(${T_NAME}
  #   ${GTEST_LIBRARIES}
  #   ${MPPI_HEADER_LIBRARY_NAME}
  # )
  add_executable(${T_NAME} ${PROJECT_SOURCE_DIR}/test/test_main.cpp ${T_FILE})
  target_link_libraries(${T_NAME}
          ${GTEST_LIBRARIES}
          ${MPPI_HEADER_LIBRARY_NAME})
  if (NOT ament_cmake_FOUND)
    gtest_add_tests(TARGET ${T_NAME})
  endif()
  # ament_add_gtest(TARGET ${T_NAME})
  set_target_properties(${T_NAME} PROPERTIES FOLDER test)
endforeach()
