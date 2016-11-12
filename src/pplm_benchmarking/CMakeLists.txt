add_executable(pplm_benchmarker.exe         pplm_benchmarker.cpp )
target_link_libraries(pplm_benchmarker.exe  ${catkin_LIBRARIES})
add_dependencies(pplm_benchmarker.exe        ${people_recognition_vision_EXPORTED_TARGETS})
