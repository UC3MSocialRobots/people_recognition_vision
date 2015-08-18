catkin_add_gtest(gtest_face_recognizer
  gtest_face_recognizer.cpp
  )
target_link_libraries(gtest_face_recognizer string_utils xml_document ${catkin_LIBRARIES})

catkin_add_gtest(gtest_height_detector
  gtest_height_detector.cpp
  )
target_link_libraries(gtest_height_detector string_utils)

# http://wiki.ros.org/rosbuild/CMakeLists#catkin_add_gtest
# Give a slow test 3 minutes (180 seconds) to complete
catkin_add_gtest(gtest_breast_detector
  gtest_breast_detector.cpp
  )

target_link_libraries(gtest_breast_detector string_utils ${PCL_LIBRARIES})

catkin_add_gtest(gtest_person_histogram
  gtest_person_histogram.cpp
  )
target_link_libraries(gtest_person_histogram string_utils lap disjoint_sets2)


catkin_add_gtest(gtest_person_histogram_set
  gtest_person_histogram_set.cpp

  )
target_link_libraries(gtest_person_histogram_set string_utils lap disjoint_sets2)


catkin_add_gtest(gtest_hist_tracking_skill
  gtest_hist_tracking_skill.cpp
  )
target_link_libraries(gtest_hist_tracking_skill lap  disjoint_sets2)

###### PPLM

catkin_add_gtest(gtest_euclidean_pplm
  gtest_euclidean_pplm.cpp
  )
target_link_libraries(gtest_euclidean_pplm lap)

catkin_add_gtest(gtest_face_recognizer_pplm
  gtest_face_recognizer_pplm.cpp
  )
target_link_libraries(gtest_face_recognizer_pplm lap)

catkin_add_gtest(gtest_height_pplm
  gtest_height_pplm.cpp)
target_link_libraries(gtest_height_pplm lap)

catkin_add_gtest(gtest_nite_pplm
  gtest_nite_pplm.cpp
  )
target_link_libraries(gtest_nite_pplm lap disjoint_sets2)

catkin_add_gtest(gtest_phs_pplm
  gtest_phs_pplm.cpp
  )
target_link_libraries(gtest_phs_pplm lap disjoint_sets2)

######

catkin_add_gtest(gtest_ppl_gating
  gtest_ppl_gating.cpp
  )
target_link_libraries(gtest_ppl_gating  gsl blas lap)

#it can be slow, give 3 minutes
catkin_add_gtest(gtest_ukf_multimodal
  gtest_ukf_multimodal.cpp
   )
target_link_libraries(gtest_ukf_multimodal  gsl blas lap)
