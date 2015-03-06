/*!
  \file        face_recognizer_pplm.h
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2014/2/3

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

A PPLMatcherTemplate using the color of the user as a matcher.

\section Parameters, subscriptions, publications
  None

\section Services
  - \b "~match_ppl"
        [people_msgs/MatchPPL]
        Match a detected PPL against a reference one.
 */
#ifndef FACE_RECOGNIZER_PPLM_H
#define FACE_RECOGNIZER_PPLM_H

// #define DISPLAY

#include <templates/pplm_template.h>
#include <image_utils/opencv_face_detector.h>
#include <face_recognizer/face_recognizer.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <ppl_utils/ppl_attributes.h>
#include <ppl_utils/people_pose_list_utils.h>

class FaceRecognizerPPLM : public PPLMatcherTemplate {
public:
  static const double MATCH_COST = 0, DEFAULT_DIAG_COST = .4,
  DEFAULT_COST = .5, NOMATCH_COST = 1;

  FaceRecognizerPPLM() : PPLMatcherTemplate("FACE_RECOGNIZER_PPLM_START", "FACE_RECOGNIZER_PPLM_STOP") {
    _classifier = image_utils::create_face_classifier();
    _nh_private.param("resize_max_width", _resize_max_width, 320);
    _nh_private.param("resize_max_height", _resize_max_height, 240);
    _nh_private.param("scale_factor", _scale_factor, 1.1);
    _nh_private.param("min_neighbors", _min_neighbors, 1);
    _nh_private.param("min_width", _min_width, 10);
    std::string face_reco_xml_file = "";
    _nh_private.param("face_reco_xml_file", face_reco_xml_file, face_reco_xml_file);
    if (!_face_reco.from_xml_file(face_reco_xml_file)) {
      ROS_FATAL("FaceRecognizerPPLM: you must specify an XML face file in parameters! "
                "For instance using _face_reco_xml_file:=$(find mypkg)/data/index.xml");
      ros::shutdown();
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  /*! \arg pp not const because we store the result of the face reco when succesful
    */
  bool pp2reco(const PP & pp, face_recognition::PersonName & ans,
               people_msgs::PeoplePoseAttributes & added_attributes) {
    ans = people_msgs::PeoplePose::RECOGNITION_FAILED;
    if (pp.rgb.width == 0 || pp.rgb.height == 0) {
      printf("FaceRecognizerPPLM: PP has no rgb image\n");
      return false;
    }
    boost::shared_ptr<void const> tracked_object;
    try {
      _rgb_bridge = cv_bridge::toCvShare(pp.rgb, tracked_object,
                                         sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
      printf("FaceRecognizerPPLM: cv_bridge exception: %s", e.what());
      return false;
    }
    //cv::imshow("rgb", _rgb_bridge->image); cv::waitKey(0);
    std::vector<cv::Rect> faces_roi;
    image_utils::detect_with_opencv
        (_rgb_bridge->image, _classifier,
         _small_img, faces_roi,
         _resize_max_width, _resize_max_height, _scale_factor,
         _min_neighbors, _min_width);
    if (faces_roi.empty()) { // no face found
      DEBUG_PRINT("FaceRecognizerPPLM::pp2reco(): no face found\n");
      return false;
    }
    cv::Mat3b face_color = _rgb_bridge->image(faces_roi.front());
    //cv::imshow("face_color", face_color); cv::waitKey(0);
    ans = _face_reco.predict_non_preprocessed_face(face_color);
    if (ans == face_recognition::NOBODY || ans == people_msgs::PeoplePose::RECOGNITION_FAILED) {
      printf("FaceRecognizerPPLM::pp2reco(): predict_non_preprocessed_face() failed\n");
      ans = people_msgs::PeoplePose::RECOGNITION_FAILED;
      return false;
    }
    // save result of face reco - only if success
    //printf("FaceRecognizerPPLM::pp2reco(): reco result='%s'\n", ans.c_str());
    added_attributes.names.push_back("face_name");
    added_attributes.values.push_back(ans);
    return true;
  } // end pp2reco()

  //////////////////////////////////////////////////////////////////////////////

  bool match(const PPL & new_ppl, const PPL & tracks, std::vector<double> & costs,
             std::vector<people_msgs::PeoplePoseAttributes> & new_ppl_added_attributes,
             std::vector<people_msgs::PeoplePoseAttributes> & tracks_added_attributes) {
    unsigned int npps = new_ppl.poses.size(),
        ntracks = tracks.poses.size();
    DEBUG_PRINT("FaceRecognizerPPLM::match(%i new PP, %i tracks)\n",
                npps, ntracks);
    // if there is only one track and one user, skip computation
    if (ntracks == 1 && npps == 1) {
      costs.clear();
      costs.resize(1, 0);
      return true;
    }

    // retrieve result of face recos in tracks
    std::vector<face_recognition::PersonName> ppl_recos(npps), track_recos(ntracks);
    for (unsigned int track_idx = 0; track_idx < ntracks; ++track_idx) {
      const PP* track = &(tracks.poses[track_idx]);
      DEBUG_PRINT("track #%i:'%s'\n", track_idx, ppl_utils::pp2string(*track).c_str());
      // retrieve result of face recos in track attributes
      if (ppl_utils::get_attribute_readonly(*track, "face_name", track_recos[track_idx]))
        continue;
      // otherwise try and detect them
      if (pp2reco(*track, track_recos[track_idx], tracks_added_attributes[track_idx]))
        continue;
      track_recos[track_idx] = people_msgs::PeoplePose::RECOGNITION_FAILED;
      DEBUG_PRINT("Could not recognize the face of track #%i\n", track_idx);
    } // end for (track_idx)
    // apply face reco on PP
    for (unsigned int pp_idx = 0; pp_idx < npps; ++pp_idx) {
      const PP* pp = &(new_ppl.poses[pp_idx]);
      if (pp2reco(*pp, ppl_recos[pp_idx], new_ppl_added_attributes[pp_idx]))
        continue;
      ppl_recos[pp_idx] = people_msgs::PeoplePose::RECOGNITION_FAILED;
      DEBUG_PRINT("Could not recognize the face of PP #%i\n", pp_idx);
    } // end for (pp_idx)

    // fill cost matrix using face reco results, must be normalized in [0-1]
    costs.resize(ntracks * npps, DEFAULT_COST);
    for (unsigned int i = 0; i < std::min(npps, ntracks); ++i) // set diagonal costs
      costs[i * ntracks + i] = DEFAULT_DIAG_COST;

    for (unsigned int pp_idx = 0; pp_idx < npps; ++pp_idx) {
      face_recognition::PersonName curr_name = ppl_recos[pp_idx];
      if (curr_name.empty()
          || curr_name == people_msgs::PeoplePose::RECOGNITION_FAILED
          || curr_name == people_msgs::PeoplePose::NO_RECOGNITION_MADE)
        continue;
      for (unsigned int track_idx = 0; track_idx < ntracks; ++track_idx) {
        face_recognition::PersonName track_name = track_recos[track_idx];
        DEBUG_PRINT("FaceRecognizerPPLM: curr:%i='%s', track:%i='%s'\n",
                    pp_idx, curr_name.c_str(), track_idx, track_name.c_str());
        if (track_name.empty()
            || track_name == people_msgs::PeoplePose::RECOGNITION_FAILED
            || track_name == people_msgs::PeoplePose::NO_RECOGNITION_MADE)
          continue;
        int cost_idx = pp_idx * ntracks + track_idx;
        
        const int MATCH_COST_auxConst = MATCH_COST;
        const int NOMATCH_COST_auxConst = NOMATCH_COST;
        
        costs[cost_idx] = (curr_name == track_name ? MATCH_COST_auxConst : NOMATCH_COST_auxConst);
      } // end for (track_idx)
    } // end for (pp_idx)
    return true;
  } // end match()
private:
  //! the redim image
  cv::Mat3b _small_img;
  //! the classifier
  cv::CascadeClassifier _classifier;
  int _resize_max_width, _resize_max_height, _min_neighbors, _min_width;
  double _scale_factor;

  cv_bridge::CvImageConstPtr _rgb_bridge;
  face_recognition::FaceRecognizer _face_reco;
}; // end class FaceRecognizerPPLM

#endif // FACE_RECOGNIZER_PPLM_H
