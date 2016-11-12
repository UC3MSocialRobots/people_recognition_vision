add_executable(artoolkit_pplm.exe artoolkit_pplm.cpp )
target_link_libraries(artoolkit_pplm.exe ${catkin_LIBRARIES})
add_dependencies(artoolkit_pplm.exe        ${people_recognition_vision_EXPORTED_TARGETS})
