add_executable(ukf_multimodal.exe ukf_multimodal.cpp  )
target_link_libraries(ukf_multimodal.exe  gsl blas lap_vision_utils  ${catkin_LIBRARIES})
add_dependencies(ukf_multimodal.exe       ${people_msgs_EXPORTED_TARGETS})

add_executable(ukf_genders.exe ukf_genders.cpp)
target_link_libraries(ukf_genders.exe gsl blas lap_vision_utils  ${catkin_LIBRARIES})
add_dependencies(ukf_genders.exe          ${people_msgs_EXPORTED_TARGETS})
