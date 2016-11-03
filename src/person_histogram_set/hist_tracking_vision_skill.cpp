/*!
  \file        hist_tracking_vision_skill.cpp
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2013/1/23

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

\class HistTrackingRgbDepthSkill
The implementation of \a HistTrackingSkill
thanks to a \a RgbDepthSkill.

The estimated positions of the users are obtained
by subscribing to a topic supplying some \a People .

\section Parameters
  - \b "~ppl_input_topic"
        [string] (default: "ppl")
        Where the face recognition results will be obtained.

\section Subscriptions
  - \b ${ppl_input_topic}
        [people_msgs::People]
        The found faces ROIs and the name of the persons recognized

 */
// vision
#include "vision_utils/rgb_depth_skill.h"
#include <people_msgs/People.h>


// people_msgs
#include "people_recognition_vision/hist_tracking_skill.h"

class HistTrackingRgbDepthSkill : public RgbDepthSkill, public HistTrackingSkill {
public:
  HistTrackingRgbDepthSkill() :
    RgbDepthSkill("FOO_VISION_SKILL2_START", "FOO_VISION_SKILL2_STOP")
  {
    // get camera model
    image_geometry::PinholeCameraModel rgb_camera_model;
    vision_utils::read_camera_model_files
        (DEFAULT_KINECT_SERIAL(), _default_depth_camera_model, rgb_camera_model);
  }

  //////////////////////////////////////////////////////////////////////////////

  void create_subscribers_and_publishers() {
    ROS_WARN("create_subscribers_and_publishers()");
    // start face detection
    _face_detector_start_pub = _nh_public.advertise<std_msgs::Int16>
        ("FACE_DETECTOR_PPLP_START", 1);
    _face_detector_start_pub.publish(std_msgs::Int16());
    _etts_face_counter_start_pub = _nh_public.advertise<std_msgs::Int16>
        ("FACE_COUNT_START", 1);
    _etts_face_counter_start_pub.publish(std_msgs::Int16());

    // get the topic names
    ros::NodeHandle nh_public, nh_private("~");
    std::string ppl_input_topic = "ppl";
    nh_private.param("ppl_input_topic",
                     ppl_input_topic,
                     ppl_input_topic);
    // susbscribe to face detection results
    _ppl_sub = nh_public.subscribe
        (ppl_input_topic, 1,
         &HistTrackingRgbDepthSkill::face_reco_result_cb, this);
  } // end create_subscribers_and_publishers();

  //////////////////////////////////////////////////////////////////////////////

  void shutdown_subscribers_and_publishers() {
    ROS_WARN("shutdown_subscribers_and_publishers()");
    _ppl_sub.shutdown();
  } // end shutdown_subscribers_and_publishers();

  //////////////////////////////////////////////////////////////////////////////

  void face_reco_result_cb
  (const people_msgs::PeopleConstPtr & msg) {
    face_recs_mutex.lock();
    face_recs.clear();
    //face_recs.push_back(msg->results.rois.size());
    face_recs.push_back(*msg);
    face_recs_mutex.unlock();
  } // end face_reco_result_cb();

  //////////////////////////////////////////////////////////////////////////////

  void process_rgb_depth(const cv::Mat3b & rgb,
                         const cv::Mat1f & depth) {
    ROS_WARN("process_rgb_depth(), time since last:%g ms", timer.getTimeMilliseconds());
    timer.reset();

    if (face_recs.size() == 0) {
      ROS_INFO_THROTTLE(1, "No face recognition results received, skipping.");
      return;
    }

    face_recs_mutex.lock();
    people_msgs::People* curr_list = &face_recs.front();
    unsigned int nusers = curr_list->poses.size();

    // find histogram for each face
    curr_phset.clear();
    for (unsigned int user_idx = 0; user_idx < nusers; ++user_idx) {
      // get seed
      //vision_utils::Rect user_roi = curr_list->poses[user_idx].image_roi;
      //cv::Point seed(user_roi.x + user_roi.width / 2, user_roi.y + user_roi.height / 2);
      cv::Point seed(curr_list->poses[user_idx].images_offsetx
                     + curr_list->poses[user_idx].rgb.width / 2,
                     curr_list->poses[user_idx].images_offsety
                     + curr_list->poses[user_idx].rgb.height / 2);
      PersonLabel new_label = user_idx+1;
      if (!curr_phset.push_back(rgb, depth, seed, _default_depth_camera_model, new_label, true, false))
        continue;
    } // end loop user_idx

    // clean
    face_recs.clear();
    face_recs_mutex.unlock();

    illus();
    wait_key();
  } // end process_rgb_depth()

  //////////////////////////////////////////////////////////////////////////////

private:
  image_geometry::PinholeCameraModel _default_depth_camera_model;
  ros::Publisher _face_detector_start_pub, _etts_face_counter_start_pub;
  std::vector<people_msgs::People> face_recs;
  ros::Subscriber _ppl_sub;
  boost::mutex face_recs_mutex;
  vision_utils::Timer timer;
}; // end class HistTrackingRgbDepthSkill

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  ros::init(argc, argv, "launcher_foo_vision_skill");
  HistTrackingRgbDepthSkill skill;
  skill.check_autostart();
  ros::spin();
  return 0;
}
