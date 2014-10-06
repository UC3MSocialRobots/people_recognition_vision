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
#include <data_fusion/ukf/ukf_multimodal.h>
#include <ppl_utils/images2pp.h>
#include <euclidean_pplm/euclidean_pplm.h>
#include <ros_utils/rosmaster_alive.h>


#define ASSERT_TRUE_TIMEOUT(cond, timeout) { Timer timer; while (timer.getTimeSeconds() < timeout && !(cond)) usleep(50 * 1000); } ASSERT_TRUE(cond)

typedef people_msgs::PeoplePose PP;
typedef people_msgs::PeoplePoseList PPL;
typedef cv::Point3f Pt3f;

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
  ASSERT_TRUE_TIMEOUT(ukf.nb_ppl_input_topics() == 1, 1);
  ASSERT_TRUE_TIMEOUT(ukf.nb_available_matchers() == 1, 1);
  ukf.stop();
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, simple_feed) {
  if (!rosmaster_alive()) return;
  ros::AsyncSpinner spinner(0);
  spinner.start();
  EuclideanPPLM pplm;
  pplm.start();
  ros::NodeHandle nh_private("~");
  nh_private.setParam("ppl_matcher_services", pplm.get_match_service_name());
  UkfMultiModal ukf;
  ukf.start();
  ASSERT_TRUE_TIMEOUT(ukf.nb_available_matchers() == 1, 1);
  PPL ppl;
  ppl.method = "gtest_ukf_multimodal";
  ukf.ppl_cb(ppl);
  ukf.stop();
}

////////////////////////////////////////////////////////////////////////////////

void test_tracks_creation(unsigned int nusers,
                          double pos_error_std_dev = 0) {
  if (!rosmaster_alive()) return;
  printf("\ntest_tracks_creation(nusers:%i)\n", nusers);
  ros::AsyncSpinner spinner(0);
  spinner.start();
  EuclideanPPLM pplm;
  pplm.start();
  ros::NodeHandle nh_private("~");
  nh_private.setParam("ppl_matcher_services", pplm.get_match_service_name());
  UkfMultiModal ukf;
  ukf.start();
  ASSERT_TRUE_TIMEOUT(ukf.nb_available_matchers() == 1, 1);
  unsigned int niters = 10, iter = 0;
  PPL ppl;
  ppl.method = "gtest_ukf_multimodal";
  ppl.header.frame_id = ukf.get_static_frame_id();
  std::vector<Pt3f> exp_users_pos;
  for (unsigned int user_idx = 0; user_idx < nusers; ++user_idx)
    //exp_users_pos.push_back(Pt3f (user_idx, user_idx, 1));
    exp_users_pos.push_back(Pt3f (cos(user_idx*M_PI*2/nusers),
                                  sin(user_idx*M_PI*2/nusers), 1));

  while (iter++ < niters) {
    ppl.header.stamp = ros::Time::now();
    ppl.poses.clear();
    for (unsigned int user_idx = 0; user_idx < nusers; ++user_idx) {
      PP pp;
      pp.header = ppl.header;
      pp.head_pose.position.x = exp_users_pos[user_idx].x
                                + combinatorics_utils::rand_gaussian() * pos_error_std_dev;
      pp.head_pose.position.y = exp_users_pos[user_idx].y
                                + combinatorics_utils::rand_gaussian() * pos_error_std_dev;
      pp.head_pose.position.z = exp_users_pos[user_idx].z
                                + combinatorics_utils::rand_gaussian() * pos_error_std_dev;
      pp.confidence = 1;
      ppl.poses.push_back(pp);
    } // end loop user_idx
    ukf.ppl_cb(ppl);
    if (ukf.nusers() == nusers)
      break;
    ASSERT_TRUE(ukf.nblobs() + ukf.nusers() == nusers)
        << "blobs:" << ukf.nblobs() << ", users:" << ukf.nusers();
    usleep(100 * 1000);
  } // end for (iter)

  // check the created tracks
  ASSERT_TRUE(ukf.nblobs() == 0 && ukf.nusers() == nusers)
      << "blobs:" << ukf.nblobs() << ", users:" << ukf.nusers();
  std::vector<Pt3f> users_pos = ukf.get_user_positions<Pt3f>();
  ASSERT_TRUE(users_pos.size() == nusers);
  for (unsigned int user_idx = 0; user_idx < nusers; ++user_idx) {
    double best_dist = std::numeric_limits<double>::max();
    unsigned int best_track_idx = 1;
    for (unsigned int track_idx = 0; track_idx < nusers; ++track_idx) {
      double dist = geometry_utils::distance_points3(users_pos[track_idx],
                                                     exp_users_pos[user_idx]);
      if (best_dist <= dist)
        continue;
      best_dist = dist;
      best_track_idx = track_idx;
    } // end for (track_idx)
    ASSERT_TRUE(best_dist <= 5 * pos_error_std_dev)
        << "user " << user_idx << ", users_pos:" << users_pos[best_track_idx]
           << ", expected:" << exp_users_pos[user_idx] << ", best_dist:" <<best_dist;
  } // end for (user_idx)
  ukf.stop();
}

TEST(TestSuite, tracks_creation_no_error) {
  for (unsigned int nusers = 0; nusers < 10; ++nusers)
    test_tracks_creation(nusers);
}

TEST(TestSuite, tracks_creation_small_error) {
  for (unsigned int nusers = 0; nusers < 10; ++nusers)
    test_tracks_creation(nusers, 1E-2);
}

TEST(TestSuite, tracks_creation_bigger_error) {
  for (unsigned int nusers = 0; nusers < 10; ++nusers)
    test_tracks_creation(nusers, 1E-1);
}

////////////////////////////////////////////////////////////////////////////////

void test_tracks_creation_destruction(unsigned int nusers) {
  if (!rosmaster_alive()) return;
  printf("\ntest_tracks_creation_destruction(nusers:%i)\n", nusers);
  ros::AsyncSpinner spinner(0);
  spinner.start();
  EuclideanPPLM pplm;
  pplm.start();
  ros::NodeHandle nh_private("~");
  nh_private.setParam("ppl_matcher_services", pplm.get_match_service_name());
  UkfMultiModal ukf;
  ukf.start();
  ASSERT_TRUE_TIMEOUT(ukf.nb_available_matchers() == 1, 1);
  unsigned int niters = 10, iter = 0;
  PPL ppl;
  ppl.method = "gtest_ukf_multimodal";
  ppl.header.frame_id = ukf.get_static_frame_id();
  std::vector<Pt3f> exp_users_pos;
  for (unsigned int user_idx = 0; user_idx < nusers; ++user_idx)
    //exp_users_pos.push_back(Pt3f (user_idx, user_idx, 1));
    exp_users_pos.push_back(Pt3f (cos(user_idx*M_PI*2/nusers),
                                  sin(user_idx*M_PI*2/nusers), 1));

  for (unsigned int loop_idx = 0; loop_idx < 2; ++loop_idx) {
    while (iter++ < niters) {
      ppl.header.stamp = ros::Time::now();
      ppl.poses.clear();
      for (unsigned int user_idx = 0; user_idx < nusers; ++user_idx) {
        PP pp;
        pp.header = ppl.header;
        pt_utils::copy3(exp_users_pos[user_idx], pp.head_pose.position);
        pp.confidence = 1;
        ppl.poses.push_back(pp);
      } // end loop user_idx
      ukf.ppl_cb(ppl);
      if (ukf.nusers() == nusers)
        break;
      ASSERT_TRUE(ukf.nblobs() + ukf.nusers() == nusers)
          << "blobs:" << ukf.nblobs() << ", users:" << ukf.nusers();
      usleep(100 * 1000);
    } // end for (iter)

    // check the created tracks
    ASSERT_TRUE(ukf.nblobs() == 0 && ukf.nusers() == nusers)
        << "blobs:" << ukf.nblobs() << ", users:" << ukf.nusers();

    // now remove the tracks
    Timer timer;
    while (timer.getTimeSeconds() < ppl_gating::DEFAULT_TRACK_TIMEOUT + 1) {
      ppl.header.stamp = ros::Time::now();
      ppl.poses.clear();
      ukf.ppl_cb(ppl);
      if (ukf.nusers() == 0)
        break;
      usleep(100 * 1000);
    } // end loop iter
    ASSERT_TRUE(ukf.nblobs() == 0 && ukf.nusers() == 0)
        << "blobs:" << ukf.nblobs() << ", users:" << ukf.nusers();
  } // end for loop_idx

  ukf.stop();
}

TEST(TestSuite, tracks_creation_destruction_no_error) {
  for (unsigned int nusers = 0; nusers < 5; ++nusers)
    test_tracks_creation_destruction(nusers);
  //test_tracks_creation_destruction(4);
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv){
  ros::init(argc, argv, "gtest_ukf_multimodal");
  // Run all the tests that were declared with TEST()
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
