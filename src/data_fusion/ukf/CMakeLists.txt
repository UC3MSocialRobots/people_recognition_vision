add_maggie_test(ukf_multimodal.exe
    SOURCES     ukf_multimodal.cpp ukf_multimodal.h
                ukf_person_pose.h
                ppl_gating.h
    LINK        gsl blas lap string_utils
)

add_maggie_test(ukf_genders.exe
    SOURCES     ukf_genders.cpp
    LINK        gsl blas lap string_utils
)

