add_maggie_test(test_conditional_particle_filter.exe
    SOURCES     test_conditional_particle_filter.cpp
                conditional_particle_filter.h
                conditional_particle_filter_laser.h
                conditional_particle_filter_laser_ros.h
    LINK        boost_signals
)

################################################################################

