add_executable(ukf_multimodal.exe ukf_multimodal.cpp ukf_multimodal.h
								ukf_person_pose.h
								ppl_gating.h)
target_link_libraries(ukf_multimodal.exe gsl blas lap string_utils ${catkin_LIBRARIES})



add_executable(ukf_genders.exe ukf_genders.cpp)
target_link_libraries(ukf_genders.exe gsl blas lap string_utils ${catkin_LIBRARIES})
