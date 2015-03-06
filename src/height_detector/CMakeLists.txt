add_executable(breast_detector.exe 
				breast_detector.cpp  breast_detector.h)
target_link_libraries(breast_detector.exe 
						${catkin_LIBRARIES} ${OpenCV_LIBS})
#add_dependencies(breast_detector.exe	template_regression.m ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(height_pplm.exe 
				height_pplm.cpp  height_pplm.h height_detector.h)
target_link_libraries(height_pplm.exe 
						${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(height_benchmark.exe 
				height_benchmark.cpp )
target_link_libraries(height_benchmark.exe 
						${catkin_LIBRARIES} ${OpenCV_LIBS})

