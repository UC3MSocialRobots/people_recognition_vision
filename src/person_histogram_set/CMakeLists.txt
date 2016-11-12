add_executable(hist_tracking_nite_skill.exe hist_tracking_nite_skill.cpp )
target_link_libraries(hist_tracking_nite_skill.exe   ${catkin_LIBRARIES}  ${PCL_LIBRARIES})
add_dependencies(hist_tracking_nite_skill.exe        ${people_recognition_vision_EXPORTED_TARGETS})

add_executable(hist_tracking_vision_skill.exe hist_tracking_vision_skill.cpp )
target_link_libraries(hist_tracking_vision_skill.exe   ${catkin_LIBRARIES}  ${PCL_LIBRARIES})
add_dependencies(hist_tracking_vision_skill.exe        ${people_recognition_vision_EXPORTED_TARGETS})

add_executable(test_person_histogram_set.exe test_person_histogram_set.cpp )
target_link_libraries(test_person_histogram_set.exe   ${catkin_LIBRARIES}  ${PCL_LIBRARIES})
add_dependencies(test_person_histogram_set.exe        ${people_recognition_vision_EXPORTED_TARGETS})

add_executable(phs_pplm.exe phs_pplm.cpp)
target_link_libraries(phs_pplm.exe   ${catkin_LIBRARIES}  ${PCL_LIBRARIES})
add_dependencies(phs_pplm.exe        ${people_recognition_vision_EXPORTED_TARGETS})
