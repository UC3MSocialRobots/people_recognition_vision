add_executable(hist_tracking_nite_skill.exe hist_tracking_nite_skill.cpp  
										    hist_tracking_skill.h)
target_link_libraries(hist_tracking_nite_skill.exe lap disjoint_sets2)

add_executable(hist_tracking_vision_skill.exe hist_tracking_vision_skill.cpp  
											  hist_tracking_skill.h)
target_link_libraries(hist_tracking_vision_skill.exe lap disjoint_sets2)

add_executable(test_person_histogram_set.exe test_person_histogram_set.cpp
										     person_histogram_set.h)
target_link_libraries(test_person_histogram_set.exe  string_utils lap disjoint_sets2)

add_executable(phs_pplm.exe phs_pplm.cpp phs_pplm.h)
target_link_libraries(phs_pplm.exe lap disjoint_sets2)
