/*!
  \file        nite_pplm.h
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2014/12/3

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

\section Parameters, subscriptions, publications
  None

\section Services
  - \b "~match_ppl"
        [people_recognition_vision/MatchPPL]
        Match a detected PPL against a reference one.
 */

#ifndef NITE_PPLM_H
#define NITE_PPLM_H

#include "people_recognition_vision/pplm_template.h"
#include "vision_utils/ppl_attributes.h"
#include "ros/ros.h"
#include <ros/package.h>

class NitePPLM : public PPLMatcherTemplate {
public:
  static const double MATCH_COST = 0, DEFAULT_DIAG_COST = .4, DEFAULT_COST = .5, NOMATCH_COST = 1;

  NitePPLM() : PPLMatcherTemplate("NITE_PPLM_START", "NITE_PPLM_STOP") {
  }

  //////////////////////////////////////////////////////////////////////////////

  bool match(const PPL & new_ppl, const PPL & tracks, std::vector<double> & costs,
             std::vector<std::string> & new_ppl_added_tagnames,
                     std::vector<std::string> & new_ppl_added_tags,
                     std::vector<unsigned int> & new_ppl_added_indices,
             std::vector<std::string> & tracks_added_tagnames,
                     std::vector<std::string> & tracks_added_tags,
                     std::vector<unsigned int> & tracks_added_indices) {
    unsigned int npps = new_ppl.people.size(),  ntracks = tracks.people.size();
    //printf("NitePPLM:match(%i tracks, %i PPs)\n", ntracks, npps);
    costs.resize(ntracks * npps, DEFAULT_COST);
    for (unsigned int i = 0; i < std::min(npps, ntracks); ++i) // set diagonal costs
      costs[i * ntracks + i] = DEFAULT_DIAG_COST;
    if (npps == 0 || ntracks == 0)
      return true;
    // if there is only one track and one user, skip computation
    if (ntracks == 1 && npps == 1) {
      costs.clear();
      costs.resize(1, 0);
      return true;
    }
    std::string curr_name,track_name;
    for (unsigned int curr_idx = 0; curr_idx < npps; ++curr_idx) {
      if (!vision_utils::get_tag
          (new_ppl.people[curr_idx], "user_multimap_name", curr_name)) {
        curr_name = "NOREC";
      }
      if (curr_name.empty()
          || curr_name == "RECFAIL"
          || curr_name == "NOREC")
        continue;
      // check matches
      for (unsigned int track_idx = 0; track_idx < ntracks; ++track_idx) {
        if (!vision_utils::get_tag
            (tracks.people[track_idx], "user_multimap_name", track_name))
          track_name = "NOREC";
        //printf("curr:%i='%s', track:%i='%s'\n", curr_idx, curr_name.c_str(), track_idx, track_name.c_str());
        if (track_name.empty()
            || track_name == "RECFAIL"
            || track_name == "NOREC")
          continue;
        int cost_idx = curr_idx * ntracks + track_idx;

        const int MATCH_COST_auxConst = MATCH_COST;
        const int NOMATCH_COST_auxConst = NOMATCH_COST;

        costs[cost_idx] = (curr_name == track_name ? MATCH_COST_auxConst : NOMATCH_COST_auxConst);
      } // end for track_idx
    } // end for curr_idx
    return true;
  } // end match()
}; // end class NitePPLM

#endif // NITE_PPLM_H
