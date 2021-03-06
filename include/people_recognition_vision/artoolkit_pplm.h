/*!
  \file        artoolkit_pplm.h
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2014/2/4

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
A matcher for PPL based on the ARToolkit distance between
the tracks and the PPL detections.

\section Parameters
  None

\section Subscriptions, publications
  None

\section Services
  - \b "~match_ppl"
        [people_recognition_vision/MatchPPL]
        Match a detected PPL against a reference one.
 */

#ifndef ARTOOLKIT_PPLM_H
#define ARTOOLKIT_PPLM_H

#include "people_recognition_vision/pplm_template.h"

class ARToolkitPPLM : public PPLMatcherTemplate {
public:
  static const double HIGH_COST = 1;

  ARToolkitPPLM() : PPLMatcherTemplate("ARTOOLKIT_PPLM_START", "ARTOOLKIT_PPLM_STOP") {
  }

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
    std::string method  = vision_utils::get_method(new_ppl);
    if (method != "artoolkit") {
      DEBUG_PRINT("ARToolkitPPLM::match() with method '%s'!='artoolkit'",
                  method.c_str());
      return false;
    }
    DEBUG_PRINT("ARToolkitPPLM::match(%i new PP, %i tracks)\n",
                ncurr_users, ntracks);
    // if there is only one track and one user, skip computation
    if (ntracks == 1 && ncurr_users == 1) {
      costs.clear();
      costs.resize(1, 0);
      return true;
    }
    costs.resize(ncurr_users * ntracks, HIGH_COST);
    for (unsigned int curr_idx = 0; curr_idx < ncurr_users; ++curr_idx) {
      std::string curr_name = new_ppl.people[curr_idx].name;
      for (unsigned int track_idx = 0; track_idx < ntracks; ++track_idx) {
        std::string track_name = tracks.people[curr_idx].name;
        if (curr_name != track_name)
          continue;
        // set a low cost
        int cost_idx = curr_idx * ntracks + track_idx;
        costs[cost_idx] = 0;
      } // end loop track_idx
    } // end loop curr_detec_idx
    return true;
  }
private:
}; // end class ARToolkitPPLM

#endif // ARTOOLKIT_PPLM_H
