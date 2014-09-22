add_maggie_test(hist_tracking_nite_skill.exe
  SOURCES       hist_tracking_nite_skill.cpp  hist_tracking_skill.h
  LINK           lap disjoint_sets2
)

add_maggie_test(hist_tracking_vision_skill
  SOURCES       hist_tracking_vision_skill.cpp  hist_tracking_skill.h
  LINK           lap disjoint_sets2
)

add_maggie_test(test_person_histogram_set.exe
  SOURCES       test_person_histogram_set.cpp
                person_histogram_set.h
  LINK          string_utils lap disjoint_sets2
)

add_maggie_test(phs_pplm.exe
  SOURCES       phs_pplm.cpp phs_pplm.h
  LINK          lap disjoint_sets2
)
