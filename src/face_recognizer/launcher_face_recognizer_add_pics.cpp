/*!
  \file        launcher_face_recognizer_add_pics.cpp
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2012/5/29

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

\class FaceRecognizerAddPics
\todo Description of the file

\section Parameters
  - \b "~ppl_input_topic"
        [string] (default: "ppl")
        Where the face detection results can be otained.

  - \b "~xml_filename"
        [string] (default: "index.xml")
        The XML filename containing the model and the pictures pathes.

  - \b "~xml_filename_out"
        [string] (default: "index_out.xml")
        The XML filename containing the model and the pictures pathes.

  - \b "~person_name"
        [string] (default: "new_user_${timestamp}")
        The name of the person that will be seen.

\section Subscriptions
  - \b ${ppl_input_topic}
        [people_msgs::PeoplePoseList]
        The images of the found faces, and their ROIs

\section Publications


 */

// ros
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
// ad_core
#include <vision_utils/skill_templates/nano_skill.h>
// vision
#include <vision_utils/image_utils/make_opencv_interface.h>
#include <vision_utils/image_utils/drawing_utils.h>
// people_msgs
#include "face_recognizer.h"
#include "people_msgs/PeoplePoseList.h"

class FaceRecognizerAddPics : public NanoSkill {
public:
  FaceRecognizerAddPics()
    : NanoSkill("FACE_RECOGNIZER_ADD_PICS_START", "FACE_RECOGNIZER_ADD_PICS_STOP") {
    // get the topic names
    _ppl_input_topic = "ppl";
    _nh_private.param("ppl_input_topic",
                     _ppl_input_topic,
                     _ppl_input_topic);
  } // end ctor

  //////////////////////////////////////////////////////////////////////////////

  void create_subscribers_and_publishers() {
    maggieDebug2("create_subscribers_and_publishers()");
    // load the model
    ros::NodeHandle nh_private("~");
    std::string xml_filename;
    nh_private.param("xml_filename", xml_filename, xml_filename);
    _face_recognizer.from_xml_file(xml_filename);
    nh_private.param("xml_filename_out", _xml_filename_out, _xml_filename_out);
    nh_private.param("person_name", _person_name, _person_name);

    // make subscriber to face detection results
    ros::NodeHandle nh_public;
    _ppl_sub = nh_public.subscribe
        (_ppl_input_topic, 1,
         &FaceRecognizerAddPics::face_detec_result_cb, this);

    // configure interface
    _window_name = "FaceRecognizerAddPics";
    cv::namedWindow(_window_name);
    cv::setMouseCallback(_window_name, mouse_cb, this);
    _can_receive_new_face = true;

    // start face reco if needed
    // start face detection
    _face_detector_start_pub = _nh_public.advertise<std_msgs::Int16>
        ("FACE_DETECTOR_PPLP_START", 1);
    _face_detector_start_pub.publish(std_msgs::Int16());
  } // end create_subscribers_and_publishers()

  //////////////////////////////////////////////////////////////////////////////

  void shutdown_subscribers_and_publishers()  {
    maggieDebug2("shutdown_subscribers_and_publishers()");
    _face_detector_start_pub.shutdown();
    _ppl_sub.shutdown();
    cvDestroyWindow(_window_name.c_str());
    cv::waitKey(5);
  } // end shutdown_subscribers_and_publishers()

  //////////////////////////////////////////////////////////////////////////////

  static void mouse_cb(int event, int x, int y, int flags, void* param) {
    //ROS_WARN("mouse_cb()");
    FaceRecognizerAddPics* this_ptr = (FaceRecognizerAddPics*) param;
    // check if a button was clicked
    if (event != cv::EVENT_LBUTTONDOWN && event != cv::EVENT_LBUTTONDBLCLK)
      return;
    int button_idx = image_utils::is_pixel_a_button
        (this_ptr->_interface_img, this_ptr->_interface_position, 2,
         this_ptr->_buttons_size, x, y);
    //ROS_WARN("mouse_cb(x:%i, y:%i, button_idx:%i)", x, y, button_idx);
    if (button_idx == 0) { // skip
      ROS_WARN("Skipping the picture");
    }
    if (button_idx == 1) { // add
      ROS_WARN("Adding the picture");
      // add the new face
      this_ptr->_face_recognizer.add_non_preprocessed_face_to_person
          (this_ptr->_current_face, this_ptr->_person_name);
      // save model
      this_ptr->_face_recognizer.to_xml_file(this_ptr->_xml_filename_out);
    }
    // clear window
    if (button_idx == 0 || button_idx == 1) {
      this_ptr->_interface_img.setTo(255);
      image_utils::draw_text_centered
          (this_ptr->_interface_img, "Waiting for a face...",
           cv::Point(this_ptr->_interface_img.cols / 2,
                     this_ptr->_interface_img.rows / 2),
           CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 0, 0));
      this_ptr->refresh_interface();
      // unlock reception of new faces
      this_ptr->_can_receive_new_face = true;
    }
  } // end mouse_cb();

  //////////////////////////////////////////////////////////////////////////////

  void face_detec_result_cb
  (const people_msgs::PeoplePoseListConstPtr & msg) {
    // do nothing if waiting for user
    if (!_can_receive_new_face)
      return;

    unsigned int nfaces = msg->poses.size();
    // do nothing if no face
    if (nfaces == 0)
      return;

    _can_receive_new_face = false;
    cv_bridge::CvImageConstPtr img_ptr;
    boost::shared_ptr<void const> tracked_object;

    for (unsigned int face_idx = 0; face_idx < nfaces; ++face_idx) {
      const people_msgs::PeoplePose* curr_pose = &(msg->poses[face_idx]);
      if (curr_pose->rgb.width == 0 || curr_pose->rgb.height == 0)
        continue;

      // try to recognize each face
      try {
        img_ptr = cv_bridge::toCvShare(curr_pose->rgb, tracked_object,
                                       sensor_msgs::image_encodings::BGR8);
      } catch (cv_bridge::Exception e) {
        ROS_WARN("cv_bridge exception:'%s'", e.what());
        continue;
      }
      img_ptr->image.copyTo(_current_face);

      std::vector<std::string> button_names;
      button_names.push_back("Skip");
      button_names.push_back("Add");
      std::vector<cv::Scalar> button_colors;
      button_colors.push_back(CV_RGB(255, 100, 100));
      button_colors.push_back(CV_RGB(100, 255, 100));
      image_utils::make_opencv_interface
          (_current_face, _interface_img, button_names, button_colors,
           _interface_position, _buttons_size);
      while (!_can_receive_new_face && is_running())
        refresh_interface();
    } // end loop face

  } // end face_detec_result_cb();

private:
  inline void refresh_interface() {
    cv::imshow(_window_name, _interface_img);
    cv::waitKey(10);
  }

  ros::Publisher _face_detector_start_pub;
  //! face detection topic
  std::string _ppl_input_topic;
  //! face detection sub
  ros::Subscriber _ppl_sub;
  //! where to save the file
  std::string _xml_filename_out;
  //! the name of the new person
  std::string _person_name;

  //! the model to recognize the faces
  bool _can_receive_new_face;
  face_recognition::FaceRecognizer _face_recognizer;
  cv::Mat3b _current_face;
  cv::Mat3b _interface_img;
  std::string _window_name;
  static const image_utils::InterfacePosition _interface_position = image_utils::LEFT;
  static const int _buttons_size = 100; // px
}; // end class FaceRecognizerAddPics

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  ros::init(argc, argv, "launcher_face_recognizer_ros");
  FaceRecognizerAddPics skill;
  ros::spin();
  return 0;
}
