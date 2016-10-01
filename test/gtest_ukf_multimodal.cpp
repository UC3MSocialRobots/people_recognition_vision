/*!
  \file        gtest_ukf_multimodal.cpp
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2014/1/31

________________________________________________________________________________

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
________________________________________________________________________________

Some tests for UkfMultiModal,
using EuclideanPPLM as a PeoplePoseListMatcher.
 */
#include <gtest/gtest.h>
#include "vision_utils/ppl_testing.h"
#include "people_recognition_vision/ukf_multimodal.h"
#include "people_recognition_vision/euclidean_pplm.h"
#include <ros_utils/rosmaster_alive.h>
#include "vision_utils/utils/string_casts_stl.h"

#define ASSERT_TRUE_TIMEOUT(cond, timeout) { Timer timer; while (timer.getTimeSeconds() < timeout && !(cond)) usleep(50 * 1000); } ASSERT_TRUE(cond)

typedef people_msgs_rl::PeoplePose PP;
typedef people_msgs_rl::PeoplePoseList PPL;
typedef geometry_utils::FooPoint3f Pt3f;

inline void assert_ntracks_eq(const UkfMultiModal & ukf, unsigned int nusers) {
  ASSERT_TRUE(ukf.nblobs() == 0 && ukf.nusers() == nusers)
      <<nusers<<" users, "<<ukf.nblobs()<<" blobs, "<<ukf.nusers()<<" ukf users";
}
inline void assert_ntracks_sumgt(const UkfMultiModal & ukf, unsigned int nusers) {
  ASSERT_TRUE(ukf.nblobs() + ukf.nusers() >= nusers)
      <<nusers<<" users, "<<ukf.nblobs()<<" blobs, "<<ukf.nusers()<<" ukf users";
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, ctor) {
  if (!rosmaster_alive()) return;
  UkfMultiModal ukf;
  ASSERT_TRUE(ukf.nusers() == 0);
  ASSERT_TRUE(ukf.nb_total_matchers() == 0);
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, connectivity_check) {
  if (!rosmaster_alive()) return;
  ros::NodeHandle nh_public, nh_private("~");
  ros::Publisher ppl_pub = nh_public.advertise<PPL>("ppl", 1);
  EuclideanPPLM pplm;
  pplm.start();
  nh_private.setParam("ppl_input_topics", ppl_pub.getTopic());
  nh_private.setParam("ppl_matcher_services", pplm.get_match_service_name());
  UkfMultiModal ukf;
  ukf.start();
  ASSERT_TRUE_TIMEOUT(ukf.nb_available_pplp() == 1, 1);
  ASSERT_TRUE_TIMEOUT(ukf.nb_available_matchers() == 1, 1);
  ukf.stop();
}

////////////////////////////////////////////////////////////////////////////////
typedef void (*UKFTestCb)(UkfMultiModal& ukf, PPL& ppl, void* c);
static const double TRACK_TIMEOUT = 1;

void test_environment_wrapper(UKFTestCb cb, void* c) {
  if (!rosmaster_alive()) return;
  ros::AsyncSpinner spinner(0);
  EuclideanPPLM pplm;
  UkfMultiModal ukf;
  PPL truth_ppl;
  spinner.start();
  ros::NodeHandle nh_private("~");
  nh_private.setParam("ppl_matcher_services", pplm.get_match_service_name());
  nh_private.setParam("track_timeout", TRACK_TIMEOUT);
  nh_private.setParam("human_walking_speed", 5.);
  ukf.start();
  pplm.start();
  truth_ppl.method = "gtest_ukf_multimodal";
  truth_ppl.header.frame_id = ukf.get_static_frame_id();
  ASSERT_TRUE_TIMEOUT(ukf.nb_available_matchers() == 1, 1);
  cb(ukf, truth_ppl, c);
  spinner.stop();
  ukf.stop();
  pplm.stop();
}

void test_simple_feed(UkfMultiModal& ukf, PPL& truth_ppl, void* c) {
  ukf.ppl_cb(truth_ppl);
} // end test_simple_feed();

TEST(TestSuite, simple_feed) {
  test_environment_wrapper(&test_simple_feed, NULL);
}

////////////////////////////////////////////////////////////////////////////////

struct ClassCreationCookie {
  unsigned int nusers;
  double pos_error_std_dev;
  ClassCreationCookie(unsigned int nusers_, double pos_error_std_dev_) :
    nusers(nusers_), pos_error_std_dev(pos_error_std_dev_) {}
};

void test_simple_track_creation(UkfMultiModal& ukf, PPL& truth_ppl, void* cookie) {
  ClassCreationCookie* c = (ClassCreationCookie*) cookie;
  printf("\ntest_simple_track_creation(%i users)\n", c->nusers);
  std::vector<Pt3f> exp_users_pos;
  for (unsigned int user_idx = 0; user_idx < c->nusers; ++user_idx)
    //exp_users_pos.push_back(Pt3f (user_idx, user_idx, 1));
    exp_users_pos.push_back(Pt3f (cos(user_idx*M_PI*2/c->nusers),
                                  sin(user_idx*M_PI*2/c->nusers), 1));

  unsigned int niters = 10, iter = 0;
  while (iter++ < niters && ukf.nusers() != c->nusers) {
    ppl_utils::ppl_factory(truth_ppl, exp_users_pos, c->pos_error_std_dev);
    ukf.ppl_cb(truth_ppl);
    usleep(100 * 1000);
    assert_ntracks_sumgt(ukf, c->nusers);
  } // end for (iter)

  // check the created tracks
  assert_ntracks_eq(ukf, c->nusers);
  ppl_utils::check_ppl_equals(truth_ppl, ukf.get_last_PPL(), c->pos_error_std_dev);
} // end run()

TEST(TestSuite, tracks_creation_no_error) {
  for (unsigned int nusers = 0; nusers < 10; ++nusers) {
    ClassCreationCookie c(nusers, 0);
    test_environment_wrapper(&test_simple_track_creation, &c);
  }
}
TEST(TestSuite, tracks_creation_small_error) {
  for (unsigned int nusers = 0; nusers < 10; ++nusers) {
    ClassCreationCookie c(nusers, 1E-2);
    test_environment_wrapper(&test_simple_track_creation, &c);
  }
}
TEST(TestSuite, tracks_creation_bigger_error) {
  for (unsigned int nusers = 0; nusers < 10; ++nusers) {
    ClassCreationCookie c(nusers, 5E-2);
    test_environment_wrapper(&test_simple_track_creation, &c);
  }
}

////////////////////////////////////////////////////////////////////////////////

void test_tracks_sudden_creation_destruction(UkfMultiModal& ukf, PPL& truth_ppl, void* cookie) {
  ClassCreationCookie* c = (ClassCreationCookie*) cookie;
  std::vector<Pt3f> exp_users_pos;
  for (unsigned int user_idx = 0; user_idx < c->nusers; ++user_idx)
    //exp_users_pos.push_back(Pt3f (user_idx, user_idx, 1));
    exp_users_pos.push_back(Pt3f (cos(user_idx*M_PI*2/c->nusers),
                                  sin(user_idx*M_PI*2/c->nusers), 1));

  for (unsigned int loop_idx = 0; loop_idx < 2; ++loop_idx) {
    // blobs creation, wait for tracks
    unsigned int niters = 10, iter = 0;
    while (iter++ < niters && ukf.nusers() != c->nusers) {
      ppl_utils::ppl_factory(truth_ppl, exp_users_pos, c->pos_error_std_dev);
      ukf.ppl_cb(truth_ppl);
      usleep(100 * 1000);
      assert_ntracks_sumgt(ukf, c->nusers);
    } // end for (iter)

    // check the created tracks
    assert_ntracks_eq(ukf, c->nusers);
    ppl_utils::check_ppl_equals(truth_ppl, ukf.get_last_PPL(), c->pos_error_std_dev);

    // now remove the tracks
    Timer timer;
    while (timer.getTimeSeconds() < TRACK_TIMEOUT + 1 && ukf.nusers() > 0) {
      ppl_utils::set_ppl_header(truth_ppl, ukf.get_static_frame_id(), ros::Time::now());
      truth_ppl.poses.clear();
      ukf.ppl_cb(truth_ppl);
      usleep(100 * 1000);
    } // end loop iter
    assert_ntracks_eq(ukf, 0);
  } // end for loop_idx
} // end test_tracks_sudden_creation_destruction();


TEST(TestSuite, tracks_sudden_creation_destruction_no_error) {
  for (unsigned int nusers = 0; nusers < 10; ++nusers) {
    ClassCreationCookie c(nusers, 0);
    test_environment_wrapper(&test_tracks_sudden_creation_destruction, &c);
  }
}
TEST(TestSuite, tracks_sudden_creation_destruction_small_error) {
  for (unsigned int nusers = 0; nusers < 10; ++nusers) {
    ClassCreationCookie c(nusers, 1E-2);
    test_environment_wrapper(&test_tracks_sudden_creation_destruction, &c);
  }
}
TEST(TestSuite, tracks_sudden_creation_destruction_bigger_error) {
  for (unsigned int nusers = 0; nusers < 10; ++nusers) {
    ClassCreationCookie c(nusers, 5E-2);
    test_environment_wrapper(&test_tracks_sudden_creation_destruction, &c);
  }
}

////////////////////////////////////////////////////////////////////////////////

void test_tracks_progresive_creation_destruction(UkfMultiModal& ukf, PPL& truth_ppl, void* cookie) {
  ClassCreationCookie* c = (ClassCreationCookie*) cookie;
  for (unsigned int iter = 1; iter <= 2 * c->nusers; ++iter) {
    bool are_users_appearing = (iter <= c->nusers);
    unsigned int nusers = (are_users_appearing ? iter : 2*c->nusers - iter);
    std::vector<Pt3f> exp_users_pos;
    for (unsigned int user_idx = 0; user_idx < nusers; ++user_idx)
      exp_users_pos.push_back(Pt3f (user_idx, user_idx, 1));
    // blobs creation, wait for tracks
    if (are_users_appearing)  { // let time for the blobs to disappear
      unsigned int niters = 10, iter = 0;
      while (iter++ < niters && ukf.nusers() != nusers) {
        ppl_utils::ppl_factory(truth_ppl, exp_users_pos, c->pos_error_std_dev);
        ukf.ppl_cb(truth_ppl);
        usleep(100 * 1000);
        assert_ntracks_sumgt(ukf, nusers);
      } // end for (iter)
    } else { // users disappearing
      Timer timer;
      while (timer.getTimeSeconds() < TRACK_TIMEOUT + 1 && ukf.nusers() != nusers) {
        ppl_utils::ppl_factory(truth_ppl, exp_users_pos, c->pos_error_std_dev);
        ukf.ppl_cb(truth_ppl);
        usleep(100 * 1000);
      } // end loop iter
    }

    // check the created tracks
    assert_ntracks_eq(ukf, nusers);
    ppl_utils::check_ppl_equals(truth_ppl, ukf.get_last_PPL(), c->pos_error_std_dev);
  } // end for nusers
} // end test_tracks_progresive_creation_destruction();

TEST(TestSuite, tracks_progressive_creation_destruction_no_error) {
  //for (unsigned int nusers = 0; nusers < 10; ++nusers) {
  ClassCreationCookie c(10, 0);
  test_environment_wrapper(&test_tracks_progresive_creation_destruction, &c);
  //}
}
TEST(TestSuite, tracks_progressive_creation_destruction_small_error) {
  //for (unsigned int nusers = 0; nusers < 10; ++nusers) {
  ClassCreationCookie c(10, 1E-2);
  test_environment_wrapper(&test_tracks_progresive_creation_destruction, &c);
  //}
}
TEST(TestSuite, tracks_progressive_creation_destruction_bigger_error) {
  //for (unsigned int nusers = 0; nusers < 10; ++nusers) {
  ClassCreationCookie c(10, 5E-2);
  test_environment_wrapper(&test_tracks_progresive_creation_destruction, &c);
  //}
}

////////////////////////////////////////////////////////////////////////////////

typedef bool (*UKFFactory)(const double time_sec,
                           PPL& truth_ppl, PPL& detection_ppl,
                           double & pos_error_std_dev);

void test_approx_following(UKFFactory factory) {
  if (!rosmaster_alive()) return;
  ros::AsyncSpinner spinner(0);
  EuclideanPPLM pplm;
  UkfMultiModal ukf;
  spinner.start();
  ros::NodeHandle nh_private("~");
  nh_private.setParam("ppl_matcher_services", pplm.get_match_service_name());
  nh_private.setParam("track_timeout", TRACK_TIMEOUT);
  ukf.start();
  pplm.start();
  ASSERT_TRUE_TIMEOUT(ukf.nb_available_matchers() == 1, 1);

  Timer timer;
  double pos_error_std_dev;
  // blobs creation, wait for tracks
  PPL truth_ppl, detection_ppl;
  truth_ppl.method = "gtest_ukf_multimodal";
  truth_ppl.header.frame_id = ukf.get_static_frame_id();
  while (factory(timer.getTimeSeconds(), truth_ppl, detection_ppl, pos_error_std_dev)) {
    ppl_utils::set_ppl_header(truth_ppl, ukf.get_static_frame_id(), ros::Time::now());
    ppl_utils::set_ppl_header(detection_ppl, ukf.get_static_frame_id(), ros::Time::now());
    unsigned int nusers = truth_ppl.poses.size();
    ukf.ppl_cb(detection_ppl);
    usleep(100 * 1000);
    assert_ntracks_sumgt(ukf, detection_ppl.poses.size());
    // check the created tracks
    if (ukf.nusers() == nusers)
      ppl_utils::check_ppl_equals(truth_ppl, ukf.get_last_PPL(), pos_error_std_dev);
  } // end for (iter)
  spinner.stop();
  ukf.stop();
  pplm.stop();
} // end test_approx_following();

////////////////////////////////////////////////////////////////////////////////

void test_random_detections_miss(UkfMultiModal& ukf, PPL& truth_ppl, void* cookie) {
  ClassCreationCookie* c = (ClassCreationCookie*) cookie;
  printf("\ntest_random_detections_miss(%i users)\n", c->nusers);
  // create truth PPL
  std::vector<Pt3f> exp_users_pos;
  for (unsigned int user_idx = 0; user_idx < c->nusers; ++user_idx)
    exp_users_pos.push_back(Pt3f (2*user_idx, 0, 1));
  ppl_utils::ppl_factory(truth_ppl, exp_users_pos, 0);

  // blobs creation, wait for tracks
  unsigned int niters = 50, iter = 0;
  PPL detection_ppl;
  while (iter++ < niters && ukf.nusers() < c->nusers) {
    ppl_utils::set_ppl_header(truth_ppl, ukf.get_static_frame_id(), ros::Time::now());
    detection_ppl.header = truth_ppl.header;
    detection_ppl.poses.clear();
    for (unsigned int user_idx = 0; user_idx < c->nusers; ++user_idx) {
      if (rand() % 4 == 0) // randomly miss detection
        continue;
      detection_ppl.poses.push_back(truth_ppl.poses[user_idx]);
      combinatorics_utils::add_gaussian_noise
          (detection_ppl.poses.back().head_pose.position, c->pos_error_std_dev);
    } // end for user_idx
    ukf.ppl_cb(detection_ppl);
    usleep(100 * 1000);
    assert_ntracks_sumgt(ukf, detection_ppl.poses.size());
  } // end for (iter)
  // check the created tracks
  assert_ntracks_eq(ukf, c->nusers);
  ppl_utils::check_ppl_equals(truth_ppl, ukf.get_last_PPL(), c->pos_error_std_dev);
} // end test_random_detections_miss();

TEST(TestSuite, random_detections_miss_no_error) {
  for (unsigned int nusers = 0; nusers < 10; ++nusers) {
    ClassCreationCookie c(nusers, 0);
    test_environment_wrapper(&test_random_detections_miss, &c);
  }
}
TEST(TestSuite, random_detections_miss_small_error) {
  for (unsigned int nusers = 0; nusers < 10; ++nusers) {
    ClassCreationCookie c(nusers, 1E-2);
    test_environment_wrapper(&test_random_detections_miss, &c);
  }
}
TEST(TestSuite, random_detections_miss_bigger_error) {
  for (unsigned int nusers = 0; nusers < 10; ++nusers) {
    ClassCreationCookie c(nusers, 5E-2);
    test_environment_wrapper(&test_random_detections_miss, &c);
  }
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv){
  ros::init(argc, argv, "gtest_ukf_multimodal");
  // Run all the tests that were declared with TEST()
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
