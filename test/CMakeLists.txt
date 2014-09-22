rosbuild_add_gtest(gtest_face_recognizer
  gtest_face_recognizer.cpp
  ../src/face_recognizer/face_recognizer.h)
target_link_libraries(gtest_face_recognizer string_utils xml_document)

rosbuild_add_gtest(gtest_head_finder
  gtest_head_finder.cpp
  ../src/height_detector/head_finder.h)

rosbuild_add_gtest(gtest_height_detector
  gtest_height_detector.cpp
  ../src/height_detector/height_detector.h)
target_link_libraries(gtest_height_detector string_utils)

# http://wiki.ros.org/rosbuild/CMakeLists#rosbuild_add_gtest
# Give a slow test 3 minutes (180 seconds) to complete
rosbuild_add_gtest(gtest_breast_detector
  gtest_breast_detector.cpp
  ../src/height_detector/breast_detector.h
                  TIMEOUT 180.0)
target_link_libraries(gtest_breast_detector string_utils ${PCL_LIBRARIES})

rosbuild_add_gtest(gtest_person_histogram
  gtest_person_histogram.cpp
  ../src/person_histogram_set/person_histogram.h)
target_link_libraries(gtest_person_histogram string_utils lap disjoint_sets2)


rosbuild_add_gtest(gtest_person_histogram_set
  gtest_person_histogram_set.cpp
  ../src/person_histogram_set/person_histogram.h
  ../src/person_histogram_set/person_histogram_set.h)
target_link_libraries(gtest_person_histogram_set string_utils lap disjoint_sets2)


rosbuild_add_gtest(gtest_hist_tracking_skill
  gtest_hist_tracking_skill.cpp
  ../src/person_histogram_set/hist_tracking_skill.h)
target_link_libraries(gtest_hist_tracking_skill lap  disjoint_sets2)

###### PPLM

rosbuild_add_gtest(gtest_euclidean_pplm
  gtest_euclidean_pplm.cpp
  ../src/euclidean_pplm/euclidean_pplm.h)
target_link_libraries(gtest_euclidean_pplm lap)

rosbuild_add_gtest(gtest_height_pplm
  gtest_height_pplm.cpp
  ../src/height_detector/height_pplm.h)
target_link_libraries(gtest_height_pplm lap)

rosbuild_add_gtest(gtest_phs_pplm
  gtest_phs_pplm.cpp
  ../src/person_histogram_set/phs_pplm.h)
target_link_libraries(gtest_phs_pplm lap disjoint_sets2)


######

#it can be slow, give 3 minutes
rosbuild_add_gtest(gtest_ukf_multimodal
  gtest_ukf_multimodal.cpp
  ../src/data_fusion/ukf/ukf_multimodal.h
                   TIMEOUT 180.0)
target_link_libraries(gtest_ukf_multimodal  gsl blas lap)
