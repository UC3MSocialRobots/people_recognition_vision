/*!
  \file        height_pplm.h
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2014/2/2

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

A PPLMatcherTemplate using the height of the users as a metric.

\section Parameters, subscriptions, publications
  None

\section Services
  - \b "~match_ppl"
        [people_recognition_vision/MatchPPL]
        Match a detected PPL against a reference one.
 */
#ifndef HEIGHT_PPLM_H
#define HEIGHT_PPLM_H

#include "people_recognition_vision/pplm_template.h"
#include "people_recognition_vision/height_detector.h"
#include "vision_utils/ppl_tags_images.h"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

class HeightPPLM : public PPLMatcherTemplate {
public:
  HeightPPLM() : PPLMatcherTemplate("HEIGHT_PPLM_START", "HEIGHT_PPLM_STOP") {
    // get camera model
    image_geometry::PinholeCameraModel rgb_camera_model;
    vision_utils::read_camera_model_files
        (vision_utils::DEFAULT_KINECT_SERIAL(), _default_depth_camera_model, rgb_camera_model);
  }

  //////////////////////////////////////////////////////////////////////////////

  bool pp2height_meter(const PP & pp, double & ans) {
    cv::Mat1f depth;
    cv::Mat1b user;
    if (!vision_utils::get_image_tag<float>(pp, "depth", depth)
        || !vision_utils::get_image_tag<uchar>(pp, "user", user)) {
      printf("HeightPPLM: PP has no depth or user image\n");
      return false;
    }
    HeightDetector::Height h = detec.height_meters
        (depth, user, _default_depth_camera_model);
    if (h.height_m == HeightDetector::ERROR)
      return false;
    ans = h.height_m;
    return true;
  } // end pp2height_meter()

  //////////////////////////////////////////////////////////////////////////////

  bool match(const PPL & new_ppl, const PPL & tracks, std::vector<double> & costs,
             std::vector<std::string> & new_ppl_added_tagnames,
                     std::vector<std::string> & new_ppl_added_tags,
                     std::vector<unsigned int> & new_ppl_added_indices,
             std::vector<std::string> & tracks_added_tagnames,
                     std::vector<std::string> & tracks_added_tags,
                     std::vector<unsigned int> & tracks_added_indices) {
    unsigned int ntracks = tracks.people.size(),
        ncurr_users = new_ppl.people.size();
    DEBUG_PRINT("HeightPPLM::match(%i new PP, %i tracks)\n",
                ncurr_users, ntracks);
    // if there is only one track and one user, skip computation
    if (ntracks == 1 && ncurr_users == 1) {
      costs.clear();
      costs.resize(1, 0);
      return true;
    }
    costs.resize(ntracks * ncurr_users, 1);

    // compute heights
    std::vector<double> new_ppl_heights(ncurr_users), track_heights(ntracks);
    for (unsigned int curr_idx = 0; curr_idx < ncurr_users; ++curr_idx) {
      if (!pp2height_meter(new_ppl.people[curr_idx], new_ppl_heights[curr_idx]))
        return false;
    } // end for (curr_idx)
    for (unsigned int track_idx = 0; track_idx < ntracks; ++track_idx) {
      if (!pp2height_meter(tracks.people[track_idx], track_heights[track_idx]))
        return false;
    } // end for (track_idx)
    DEBUG_PRINT("HeightPPLM:new_ppl_heights:%s, track_heights:%s",
                vision_utils::iterable_to_string(new_ppl_heights).c_str(),
                vision_utils::iterable_to_string(track_heights).c_str());

    // compute height differences
    for (unsigned int curr_idx = 0; curr_idx < ncurr_users; ++curr_idx) {
      for (unsigned int track_idx = 0; track_idx < ntracks; ++track_idx) {
        int cost_idx = curr_idx * ntracks + track_idx;
        double delta_size = fabs(new_ppl_heights[curr_idx] - track_heights[track_idx]);
        // convert to [0-1] - octave:fplot ("1-exp(-3*x)", [0, .5])
        double cost = 1. - exp(-delta_size * 3.);
        costs[cost_idx] = cost;
      } // end for (track_idx)
    } // end for (curr_idx)
    return true;
  }
private:
  cv_bridge::CvImageConstPtr _depth_bridge, _user_bridge;
  image_geometry::PinholeCameraModel _default_depth_camera_model;
  HeightDetector detec;
}; // end class HeightPPLM

#endif // HEIGHT_PPLM_H
