add_executable(test_conditional_particle_filter.exe 
				test_conditional_particle_filter.cpp
                conditional_particle_filter.h
                conditional_particle_filter_laser.h
                conditional_particle_filter_laser_ros.h)
target_link_libraries(test_conditional_particle_filter.exe boost_signals)
