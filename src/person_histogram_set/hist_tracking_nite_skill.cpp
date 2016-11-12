/*!
  \file        hist_tracking_nite_skill.cpp
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2012/10/16

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

\todo Description of the file

cf
https://code.ros.org/svn/ros-pkg/stacks/common_tutorials/trunk/ROS_tutorial_math/
http://www.embeddedheaven.com/ros-nodelet.htm

\section Parameters

\section Subscriptions

\section Publications

 */

// #include "skill_templates/nite/nite_ROS_subscriber_template.h"
#include "vision_utils/accessible_to_string.h"
#include "vision_utils/rgb_depth_user_skill.h"
#include "vision_utils/get_all_different_values.h"
#include "people_recognition_vision/hist_tracking_skill.h"

class HistTrackingNiteSkill : public vision_utils::RgbDepthUserSkill, public HistTrackingSkill {
public:
  HistTrackingNiteSkill() : RgbDepthUserSkill("HIST_TRACKING_NITE_SKILL_START",
                                              "HIST_TRACKING_NITE_SKILL_STOP") {
    // get camera model
    image_geometry::PinholeCameraModel rgb_camera_model;
    vision_utils::read_camera_model_files
        (vision_utils::DEFAULT_KINECT_SERIAL(), _default_depth_camera_model, rgb_camera_model);
  }

  //////////////////////////////////////////////////////////////////////////////

  virtual void create_subscribers_and_publishers() {}
  virtual void shutdown_subscribers_and_publishers() {}

  void process_rgb_depth_user(const cv::Mat3b & color,
          const cv::Mat1f & depth,
          const cv::Mat1b & user) {
    ROS_INFO_THROTTLE(1, "HistTrackingNiteSkill:fn()");
    DEBUG_PRINT("HistTrackingNiteSkill:fn()");
    // ROS_INFO("fn()");
    vision_utils::Timer timer_callback;

    // get the list of all labels in user
    std::vector<uchar> user_labels;
    vision_utils::get_all_different_values(user, user_labels, true);
    unsigned int nusers = user_labels.size();

    // for each label, compute its histogram
    curr_phset.clear();
    for (unsigned int user_label_idx = 0; user_label_idx < nusers; ++user_label_idx) {
      uchar user_label = user_labels[user_label_idx];
      // get user mask
      user_mask = (user == user_label);
      PersonLabel new_label = user_label_idx+1;
      if (!curr_phset.push_back(color, user_mask, depth, _default_depth_camera_model, new_label, true, false))
        continue;
    } // end loop user_label_idx
    vision_utils::Timer::Time time_compute_vector_of_histograms = timer_callback.time();

    if (!compare())
      return;

    say_description_sentence();

    ///// illus
#if 0
    // user_skeleton_illus
    user_image_to_rgb(user, user_skeleton_illus, 8);
    vision_utils::draw_skeleton_list(user_skeleton_illus, skeleton_list);
    cv::imshow("user_skeleton_illus", user_skeleton_illus);
#endif
    illus();
    wait_key();
    ROS_WARN_THROTTLE(1, "user_labels:'%s' (size:%i), time for "
                      "compute_vector_of_histograms(): %g ms, total callback:%g ms",
                      vision_utils::accessible_to_string(user_labels).c_str(),
                      nusers,
                      time_compute_vector_of_histograms, timer_callback.getTimeMilliseconds());

  } // end image_callback();

private:
  cv::Mat1b user_mask;
}; // end class HistTrackingNiteSkill

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  ros::init(argc, argv, "hist_tracking_nite_skill");
  HistTrackingNiteSkill skill;
  skill.start();
  ros::spin();
}
