add_executable(test_face_recognizer.exe test_face_recognizer.cpp )
target_link_libraries(test_face_recognizer.exe   ${catkin_LIBRARIES})
add_dependencies(test_face_recognizer.exe        ${people_msgs_EXPORTED_TARGETS})

add_executable(launcher_face_recognizer_ros.exe launcher_face_recognizer_ros.cpp)
target_link_libraries(launcher_face_recognizer_ros.exe  ${catkin_LIBRARIES})
add_dependencies(launcher_face_recognizer_ros.exe        ${people_msgs_EXPORTED_TARGETS})

add_executable(launcher_face_recognizer_add_pics.exe launcher_face_recognizer_add_pics.cpp)
target_link_libraries(launcher_face_recognizer_add_pics.exe  ${catkin_LIBRARIES})
add_dependencies(launcher_face_recognizer_add_pics.exe        ${people_msgs_EXPORTED_TARGETS})

add_executable(launcher_face_recognizer_viewer.exe launcher_face_recognizer_viewer.cpp)
target_link_libraries(launcher_face_recognizer_viewer.exe ${catkin_LIBRARIES})
add_dependencies(launcher_face_recognizer_viewer.exe        ${people_msgs_EXPORTED_TARGETS})

add_executable(face_recognizer_pplm.exe face_recognizer_pplm.cpp )
target_link_libraries(face_recognizer_pplm.exe  ${catkin_LIBRARIES})
add_dependencies(face_recognizer_pplm.exe        ${people_msgs_EXPORTED_TARGETS})
