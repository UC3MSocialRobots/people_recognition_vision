add_executable(test_conditional_particle_filter.exe 
				test_conditional_particle_filter.cpp
                
                
                )
target_link_libraries(test_conditional_particle_filter.exe boost_signals  ${catkin_LIBRARIES})
