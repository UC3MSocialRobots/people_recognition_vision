/*!
  \file        phs_pplm.h
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
#ifndef PHS_PPLM_H
#define PHS_PPLM_H

// #define DISPLAY

#include <templates/pplm_template.h>
#include <person_histogram_set/person_histogram.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <kinect_utils/user_image_to_rgb.h>

class PHSPPLM : public PPLMatcherTemplate {
public:
  typedef PersonHistogram    PH;

  PHSPPLM() : PPLMatcherTemplate("PHS_PPLM_START", "PHS_PPLM_STOP") {
    // get camera model
    image_geometry::PinholeCameraModel rgb_camera_model;
    kinect_openni_utils::read_camera_model_files
        (DEFAULT_KINECT_SERIAL(), _default_depth_camera_model, rgb_camera_model);
  }

  //////////////////////////////////////////////////////////////////////////////

  bool pp2phs(const PP & pp, PH & ans) {
    if (pp.rgb.width == 0 || pp.rgb.height == 0) {
      printf("PHSPPLM: PP has no rgb image\n");
      return false;
    }
    boost::shared_ptr<void const> tracked_object;
    try {
      _rgb_bridge = cv_bridge::toCvShare(pp.rgb, tracked_object,
                                         sensor_msgs::image_encodings::BGR8);
      _depth_bridge = cv_bridge::toCvShare(pp.depth, tracked_object,
                                           sensor_msgs::image_encodings::TYPE_32FC1);
      _user_bridge = cv_bridge::toCvShare(pp.user, tracked_object,
                                          sensor_msgs::image_encodings::TYPE_8UC1);
    } catch (cv_bridge::Exception& e) {
      printf("PHSPPLM: cv_bridge exception: %s", e.what());
      return false;
    }
    const cv::Mat3b & rgb = _rgb_bridge->image;
    const cv::Mat1f & depth = _depth_bridge->image;
    const cv::Mat1b & user = _user_bridge->image;
    // careful about the order here!
    bool ok = ans.create(rgb, user, depth, _default_depth_camera_model);

#ifdef DISPLAY
    cv::imshow("rgb", _rgb_bridge->image);
    cv::imshow("depth", _depth_bridge->image);
    cv::imshow("user", _user_bridge->image > 0);
    cv::imshow("illus_color_img", ans.get_illus_color_img());
    cv::imshow("illus_color_mask", ans.get_illus_color_mask() > 0);
    cv::imshow("multimask", user_image_to_rgb(ans.get_multimask()));
    ans.show_illus_image(0);
#endif // no DISPLAY
    return ok;
  } // end pp2phs()

  //////////////////////////////////////////////////////////////////////////////

  bool match(const PPL & new_ppl, const PPL & tracks, std::vector<double> & costs) {
    unsigned int ncurr_users = new_ppl.poses.size(),
        ntracks = tracks.poses.size();
    DEBUG_PRINT("PHSPPLM::match(%i new PP, %i tracks)\n",
                ncurr_users, ntracks);
    costs.resize(ntracks * ncurr_users, 1);

    // compute heights
    std::vector<PH> new_ppl_heights(ncurr_users), track_heights(ntracks);
    std::ostringstream desc;
    desc << "New PPL:";
    for (unsigned int curr_idx = 0; curr_idx < ncurr_users; ++curr_idx) {
      if (!pp2phs(new_ppl.poses[curr_idx], new_ppl_heights[curr_idx])) {
        printf("curr_idx:%i returned an error\n", curr_idx);
        return false;
      }
      desc << new_ppl_heights[curr_idx].description_sentence(curr_idx) << ",";
    } // end for (curr_idx)
    desc << ";  tracks:";
    for (unsigned int track_idx = 0; track_idx < ntracks; ++track_idx) {
      if (!pp2phs(tracks.poses[track_idx], track_heights[track_idx])) {
        printf("track_idx:%i returned an error\n", track_idx);
        return false;
      }
      desc << track_heights[track_idx].description_sentence(track_idx) << ",";
    } // end for (track_idx)

    // set diagonal costs to 0
    for (unsigned int curr_idx = 0; curr_idx < ncurr_users; ++curr_idx) {
      for (unsigned int track_idx = 0; track_idx < ntracks; ++track_idx) {
        int cost_idx = curr_idx * ntracks + track_idx;
        double delta_size = new_ppl_heights[curr_idx].compare_to(track_heights[track_idx]);
        costs[cost_idx] = delta_size;
      } // end for (track_idx)
    } // end for (curr_idx)
    ROS_INFO_THROTTLE(10, "PHSPPLM:desc:%s, costs:\n%s\n", desc.str().c_str(),
                costs_vec2string(costs, ncurr_users, ntracks).c_str());
    return true;
  }
private:
  cv_bridge::CvImageConstPtr _rgb_bridge, _depth_bridge, _user_bridge;
  image_geometry::PinholeCameraModel _default_depth_camera_model;
}; // end class PHSPPLM



#endif // PHS_PPLM_H
