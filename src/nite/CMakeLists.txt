add_executable(nite_pplm.exe nite_pplm.cpp
  ../../include/people_recognition_vision/nite_pplm.h)
target_link_libraries(nite_pplm.exe ${catkin_LIBRARIES})
add_dependencies(nite_pplm.exe        ${people_recognition_vision_EXPORTED_TARGETS})
