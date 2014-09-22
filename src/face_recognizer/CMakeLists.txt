### rules for FaceRecognizer
#add_maggie_lib( face_recognizer
#    SOURCES     face_recognizer.cpp
#    LINK        xml_document # opencv_facerec
#)

add_executable(test_face_recognizer.exe test_face_recognizer.cpp face_recognizer.h)
target_link_libraries(test_face_recognizer.exe string_utils xml_document)

add_executable(launcher_face_recognizer_ros.exe launcher_face_recognizer_ros.cpp)
target_link_libraries(launcher_face_recognizer_ros.exe xml_document)

add_executable(launcher_face_recognizer_add_pics.exe launcher_face_recognizer_add_pics.cpp)
target_link_libraries(launcher_face_recognizer_add_pics.exe xml_document)

add_executable(launcher_face_recognizer_viewer.exe launcher_face_recognizer_viewer.cpp)
#target_link_libraries(test_args_parser.exe )
