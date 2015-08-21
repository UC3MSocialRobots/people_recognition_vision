/*!
  \file        euclidean_pplm.h
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

A matcher for PPL based on the Euclidean distance between
the tracks and the PPL detections.

\section Parameters
  None

\section Subscriptions, publications
  None

\section Services
  - \b "~match_ppl"
        [people_msgs/MatchPPL]
        Match a detected PPL against a reference one.
 */

#ifndef EUCLIDEAN_PPLM_H
#define EUCLIDEAN_PPLM_H

#include "vision_utils/pplm_template.h"
#include "vision_utils/utils/distances.h"

class EuclideanPPLM : public PPLMatcherTemplate {
public:
  typedef geometry_msgs::Point Pt3;

  EuclideanPPLM() : PPLMatcherTemplate("EUCLIDEAN_PPLM_START", "EUCLIDEAN_PPLM_STOP") {
  }

  //////////////////////////////////////////////////////////////////////////////

  bool match(const PPL & new_ppl, const PPL & tracks, std::vector<double> & costs,
             std::vector<people_msgs::PeoplePoseAttributes> & new_ppl_added_attributes,
             std::vector<people_msgs::PeoplePoseAttributes> & tracks_added_attributes) {
    unsigned int ntracks = tracks.poses.size(), npps = new_ppl.poses.size();
    DEBUG_PRINT("EuclideanPPLM::match(%i new PP, %i tracks)\n",
                npps, ntracks);
    // if there is only one track and one user, skip computation
    if (ntracks == 1 && npps == 1) {
      costs.clear();
      costs.resize(1, 0);
      return true;
    }
    costs.resize(npps * ntracks);
    for (unsigned int pp_idx = 0; pp_idx < npps; ++pp_idx) {
      const PP* pp = &(new_ppl.poses[pp_idx]);
      for (unsigned int track_idx = 0; track_idx < ntracks; ++track_idx) {
        const PP* track = &(tracks.poses[track_idx]);
        double curr_dist = geometry_utils::distance_points3
                           (pp->head_pose.position, track->head_pose.position);
        // convert to [0-1] - octave:fplot ("1-exp(-3*x)", [0, .5])
        double cost = 1. - exp(-curr_dist * 3.);
        int cost_idx = pp_idx * ntracks + track_idx;
        costs[cost_idx] = cost;
      } // end loop track_idx
    } // end loop pp_detec_idx
    return true;
  }
private:
}; // end class EuclideanPPLM

#endif // EUCLIDEAN_PPLM_H
