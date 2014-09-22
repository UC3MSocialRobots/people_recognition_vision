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
  - \b "~use_gating"
        [bool] (default: true)
        Use gating?

  - \b "~gate_size"
        [double, meters] (default: 2)
        The size of the gate, in meters.

\section Subscriptions, publications
  None

\section Services
  - \b "~match_ppl"
        [people_msgs/MatchPPL]
        Match a detected PPL against a reference one.
 */

#ifndef EUCLIDEAN_PPLM_H
#define EUCLIDEAN_PPLM_H

#include <src/templates/pplm_template.h>
#include <src/geom/distances.h>

class EuclideanPPLM : public PPLMatcherTemplate {
public:
  EuclideanPPLM() : PPLMatcherTemplate("EUCLIDEAN_PPLM_START", "EUCLIDEAN_PPLM_STOP") {
    _nh_private.param("use_gating", _use_gating, true);
    double gate_size = 2;
    _nh_private.param("gate_size", gate_size, gate_size);
    _gate_size_sq = gate_size * gate_size;
  }

  //////////////////////////////////////////////////////////////////////////////

  bool match(const PPL & new_ppl, const PPL & tracks, std::vector<double> & costs) {
    unsigned int ntracks = tracks.poses.size(),
        ncurr_users = new_ppl.poses.size();
    DEBUG_PRINT("EuclideanPPLM::match(%i new PP, %i tracks)\n",
                ncurr_users, ntracks);
    costs.resize(ncurr_users * ntracks);
    for (unsigned int curr_idx = 0; curr_idx < ncurr_users; ++curr_idx) {
      geometry_msgs::Point curr_user_pos =
          new_ppl.poses[curr_idx].head_pose.position;
      for (unsigned int track_idx = 0; track_idx < ntracks; ++track_idx) {
        geometry_msgs::Point track_pos =
            tracks.poses[track_idx].head_pose.position;
        double curr_dist_sq = geometry_utils::distance_points3_squared
            (track_pos, curr_user_pos);
        // make the cost as infinite if outside of the gate
        if (_use_gating && curr_dist_sq > _gate_size_sq)
          curr_dist_sq = std::numeric_limits<double>::max() ;
        int cost_idx = curr_idx * ntracks + track_idx;
        costs[cost_idx] = curr_dist_sq;
      } // end loop track_idx
    } // end loop curr_detec_idx
    return true;
  }
private:
  double _gate_size_sq;
  bool _use_gating;
}; // end class EuclideanPPLM

#endif // EUCLIDEAN_PPLM_H
