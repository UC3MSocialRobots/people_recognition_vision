/*!
  \file        launcher_face_recognizer_viewer.cpp
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2012/5/30

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

\class FaceRecognizerViewer
This node subscribes to the face recognition results and
displays them overlaid on the camera frame.

\section Parameters
  - \b "~ppl_input_topic"
        [string] (default: "ppl")
        Where the face recognition results will be obtained.

\section Subscriptions
  - \b ${ppl_input_topic}
        [people_msgs_rl::PeoplePoseList]
        The found faces ROIs and the name of the persons recognized

\section Publications
  None.
 */
// ros
#include <ros/ros.h>
#include <boost/thread/mutex.hpp>
// utils
#include "vision_utils/utils/rect_utils.h"
#include "vision_utils/utils/multi_subscriber.h"
// vision
#include "vision_utils/rgb_skill.h"
#include "vision_utils/color_utils.h"
#include "vision_utils/drawing_utils.h"
// people_msgs_rl
#include "people_msgs_rl/PeoplePoseList.h"

class FaceRecognizerViewer : public RgbSkill {
public:
  FaceRecognizerViewer()
    : RgbSkill("FACE_RECOGNIZER_VIEWER_START", "FACE_RECOGNIZER_VIEWER_STOP") {
  } // end ctor

  //////////////////////////////////////////////////////////////////////////////

  void create_subscribers_and_publishers() {
    maggieDebug2("create_subscribers_and_publishers()");
    // get the topic names
    std::string ppl_input_topics = "ppl";
    _nh_private.param("ppl_input_topics", ppl_input_topics, ppl_input_topics);

    // subscribers
    _face_recognition_results_subs = ros::MultiSubscriber::subscribe
        (_nh_public, ppl_input_topics, 1,
         &FaceRecognizerViewer::face_reco_result_cb, this);
  } // end create_subscribers_and_publishers()

  //////////////////////////////////////////////////////////////////////////////

  void face_reco_result_cb
  (const people_msgs_rl::PeoplePoseListConstPtr & msg) {
    // do nothing if frame not ready
    if (!is_running()) {
      ROS_WARN_THROTTLE(1, "Received face recognition but not running!");
      return;
    }
    _received_rec_mutex.lock();
    _received_rec.push_back(*msg);
    _received_rec_mutex.unlock();
  } // end face_reco_result_cb();

  //////////////////////////////////////////////////////////////////////////////

  static inline long string_to_some_int(const std::string & str) {
    unsigned int str_size = str.size();
    long ans = 0;
    for (unsigned int char_idx = 0; char_idx < str_size; ++char_idx)
      ans += (unsigned int) str.at(char_idx);
    return ans;
  }

  void process_rgb(const cv::Mat3b & rgb) {
    // maggieDebug3("process_rgb()");
    rgb.copyTo(_frame_out);

    _received_rec_mutex.lock();
    // ROS_WARN("_received_rec.size():%i", _received_rec.size());
    for (unsigned int rec_idx = 0; rec_idx < _received_rec.size(); ++rec_idx) {
      people_msgs_rl::PeoplePoseList* curr_rec = &(_received_rec[rec_idx]);
      // draw rectangles
      for (unsigned int face_idx = 0; face_idx < curr_rec->poses.size(); ++face_idx) {
        const people_msgs_rl::PeoplePose* curr_pose = &(curr_rec->poses[face_idx]);
        std::string person_name = curr_pose->person_name;
        maggieDebug2("rec #%i: person #%i:'%s'",
                     rec_idx, face_idx, person_name.c_str());
        cv::Rect face_ROI;
        face_ROI.x = curr_pose->images_offsetx;
        face_ROI.y = curr_pose->images_offsety;
        face_ROI.width = curr_pose->rgb.width;
        face_ROI.height = curr_pose->rgb.height;
        // geometry_utils::copy_rectangles(curr_pose->image_roi, face_ROI);
        // cv::Scalar txt_color = color_utils::color<cv::Scalar>(face_idx);
        cv::Scalar txt_color = color_utils::color<cv::Scalar>
            (string_to_some_int(person_name));
        cv::rectangle(_frame_out, face_ROI, txt_color, 2);
        cv::Point text_pt(face_ROI.x + face_ROI.width / 2,
                          face_ROI.y + face_ROI.height + 10 + 25 * rec_idx);
        //ROS_WARN("Writing '%s' in (%i, %i)", person_name.c_str(), text_pt.x, text_pt.y);
        image_utils::draw_text_centered // white background
            (_frame_out, person_name, text_pt,
             cv::FONT_HERSHEY_DUPLEX, 1, CV_RGB(255, 255, 255), 3);
        image_utils::draw_text_centered
            (_frame_out, person_name, text_pt,
             cv::FONT_HERSHEY_DUPLEX, 1, txt_color, 2);
      } // end loop face_idx
    } // end loop rec
    _received_rec.clear();
    _received_rec_mutex.unlock();

    cv::imshow("FaceRecognizerViewer", _frame_out);
    cv::waitKey(10);
  } // end proceso()

  //////////////////////////////////////////////////////////////////////////////

  void shutdown_subscribers_and_publishers()  {
    maggieDebug2("shutdown_subscribers_and_publishers()");
    _face_recognition_results_subs.shutdown();
  } // end shutdown_subscribers_and_publishers()

  //////////////////////////////////////////////////////////////////////////////

private:
  //! face reco sub
  ros::MultiSubscriber _face_recognition_results_subs;
  std::vector<people_msgs_rl::PeoplePoseList> _received_rec;
  //! the frame where the stuff is drawn
  cv::Mat3b _frame_out;
  boost::mutex _received_rec_mutex;
}; // end class FaceRecognizerViewer

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  ros::init(argc, argv, "launcher_face_recognizer_viewer");
  FaceRecognizerViewer skill;
  skill.check_autostart();
  ros::spin();
  return 0;
}
