add_executable(ukf_multimodal.exe ukf_multimodal.cpp
  ../../../include/people_recognition_vision/ppl_gating.h
  ../../../include/people_recognition_vision/ukf_multimodal.h
)
target_link_libraries(ukf_multimodal.exe  gsl blas   ${catkin_LIBRARIES})
add_dependencies(ukf_multimodal.exe        ${people_recognition_vision_EXPORTED_TARGETS})

#add_executable(ukf_genders.exe ukf_genders.cpp)
#target_link_libraries(ukf_genders.exe gsl blas   ${catkin_LIBRARIES})
#add_dependencies(ukf_genders.exe          ${people_recognition_vision_EXPORTED_TARGETS})
