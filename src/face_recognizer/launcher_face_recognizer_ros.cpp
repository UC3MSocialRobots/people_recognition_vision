/*!
  \file        launcher_face_recognizer_ros.cpp
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2012/5/28

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

\class FaceRecognizerRos
A face recognizer.
This node needs another node to make the face detection:
it will recognizes (put a name on) the faces that the latter finds.

\section Parameters
  - \b "~ppl_input_topic"
        [string] (default: "ppl")
        Where the face detection results can be otained.

  - \b "~ppl_output_topic"
        [string] (default: "ppl_output_topic")
        Where the face recognition results will be published.

  - \b "~xml_filename"
        [string] (default: "index.xml")
        The XML filename containing the model and the pictures pathes.

\section Subscriptions
  - \b ${ppl_input_topic}
        [people_msgs::PeoplePoseList]
        The images of the found faces, and their ROIs

\section Publications
  - \b ${ppl_output_topic}
        [people_msgs::PeoplePoseList]
        The found faces ROIs and the name of the persons recognized

 */

// ros
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
// ad_core
#include "vision_utils/nano_skill.h"
// vision
// people_msgs
#include "people_recognition_vision/face_recognizer.h"
#include <people_msgs/PeoplePoseList.h>

class FaceRecognizerRos : public NanoSkill {
public:
  FaceRecognizerRos()
    : NanoSkill("FACE_RECOGNIZER_START", "FACE_RECOGNIZER_STOP") {

    // get the topic names
    _ppl_input_topic = "ppl";
    _nh_private.param("ppl_input_topic",
                     _ppl_input_topic,
                     _ppl_input_topic);

    // advertize for the updated PPL
    _face_recognition_results_pub = _nh_private.advertise
        <people_msgs::PeoplePoseList>("ppl", 1);
  } // end ctor

  //////////////////////////////////////////////////////////////////////////////

  void create_subscribers_and_publishers() {
    maggieDebug2("create_subscribers_and_publishers()");

    // start face detection
    _face_detector_start_pub = _nh_public.advertise<std_msgs::Int16>
        ("FACE_DETECTOR_PPLP_START", 1);
    _face_reco_viewer_start_pub = _nh_public.advertise<std_msgs::Int16>
        ("FACE_RECOGNIZER_VIEWER_START", 1);
    _face_detector_start_pub.publish(std_msgs::Int16());
    _face_reco_viewer_start_pub.publish(std_msgs::Int16());

    // load the model
    std::string xml_filename;
    _nh_private.param("xml_filename", xml_filename, xml_filename);
    _face_recognizer.from_xml_file(xml_filename);

    // make subscriber to face detection results
    _ppl_sub = nh_public.subscribe
        (_ppl_input_topic, 1,
         &FaceRecognizerRos::face_detec_result_cb, this);
  } // end create_subscribers_and_publishers()

  //////////////////////////////////////////////////////////////////////////////

  void shutdown_subscribers_and_publishers()  {
    maggieDebug2("shutdown_subscribers_and_publishers()");
    _face_detector_start_pub.shutdown();
    _face_reco_viewer_start_pub.shutdown();
    _ppl_sub.shutdown();
  } // end shutdown_subscribers_and_publishers()

  //////////////////////////////////////////////////////////////////////////////

  void face_detec_result_cb
  (const people_msgs::PeoplePoseListConstPtr & msg) {
    // prepair the msg to be published
    _face_recognition_results_msg = *msg;

    // try to recognize each face
    cv_bridge::CvImageConstPtr img_ptr;
    for (unsigned int face_idx = 0; face_idx < msg->poses.size(); ++face_idx) {
      const people_msgs::PeoplePose*
          curr_pose_in = &(msg->poses[face_idx]);
      people_msgs::PeoplePose*
          curr_pose_out = &(_face_recognition_results_msg.poses[face_idx]);
      //      if (curr_pose_in->has_image == false)
      //        continue;

      boost::shared_ptr<void const> tracked_object;
      try {
        img_ptr = cv_bridge::toCvShare
            (curr_pose_in->rgb, tracked_object,
             sensor_msgs::image_encodings::BGR8);
      } catch (cv_bridge::Exception e) {
        ROS_WARN("cv_bridge exception:'%s'", e.what());
        // mark the person as not recognized
        curr_pose_out->person_name = people_msgs::PeoplePose::RECOGNITION_FAILED;
        continue;
      }

      // convert the color face to B&W version
      const cv::Mat* face_color = &(img_ptr->image);

      // call the predict_preprocessed_face() function of _face_recognizer on face_color
      curr_pose_out->person_name = _face_recognizer.predict_non_preprocessed_face(*face_color);

      // show the face in a window
      //cv::imshow(person_name, *face_color);
    } // end loop face_idx

    //cv::waitKey(10);

    // publish the built message
    _face_recognition_results_pub.publish(_face_recognition_results_msg);
  } // end face_detec_result_cb();

private:
  ros::Publisher _face_detector_start_pub;
  ros::Publisher _face_reco_viewer_start_pub;

  ros::NodeHandle nh_public;
  //! face detection topic
  std::string _ppl_input_topic;
  //! face detection sub
  ros::Subscriber _ppl_sub;

  //! face reco topic
  people_msgs::PeoplePoseList _face_recognition_results_msg;
  //! face reco pub
  ros::Publisher _face_recognition_results_pub;

  //! the model to recognize the faces
  face_recognition::FaceRecognizer _face_recognizer;
}; // end class FaceRecognizerRos

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  ros::init(argc, argv, "launcher_face_recognizer_ros");
  FaceRecognizerRos skill;
  skill.check_autostart();
  ros::spin();
  return 0;
}
