### rules for FaceRecognizer
#add_maggie_lib( face_recognizer
#    SOURCES     face_recognizer.cpp
#    LINK        xml_document # opencv_facerec
#)

add_maggie_test(test_face_recognizer.exe
    SOURCES     test_face_recognizer.cpp face_recognizer.h
    LINK        string_utils xml_document
)

add_maggie_test(launcher_face_recognizer_ros.exe
    SOURCES     launcher_face_recognizer_ros.cpp
    LINK        xml_document
)

add_maggie_test(launcher_face_recognizer_add_pics.exe
    SOURCES     launcher_face_recognizer_add_pics.cpp
    LINK        xml_document
)

add_maggie_test(launcher_face_recognizer_viewer.exe
    SOURCES     launcher_face_recognizer_viewer.cpp
    LINK        
)
