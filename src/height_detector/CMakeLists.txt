add_maggie_test(breast_detector.exe
    SOURCES     breast_detector.cpp  breast_detector.h
    LINK
)
ADD_DEPENDENCIES(breast_detector.exe template_regression.m)

add_maggie_test(height_pplm.exe
    SOURCES     height_pplm.cpp  height_pplm.h height_detector.h
    LINK
)
