add_executable(euclidean_pplm.exe euclidean_pplm.cpp )
target_link_libraries(euclidean_pplm.exe ${catkin_LIBRARIES})
add_dependencies(euclidean_pplm.exe        ${people_recognition_vision_EXPORTED_TARGETS})
