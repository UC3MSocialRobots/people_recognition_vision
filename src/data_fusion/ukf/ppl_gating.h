/*!
  \file        ppl_gating.h
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2012/10/20

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

 */

#ifndef PEOPLE_POSE_LIST_GATING_H
#define PEOPLE_POSE_LIST_GATING_H

#include <combinatorics/assignment_utils.h>
#include <ppl_utils/ppl_attributes.h>
#include <ppl_utils/people_pose_list_utils.h>
#include "ukf_person_pose.h"

#ifndef DEBUG_PRINT
//#define DEBUG_PRINT(...)   {}
//#define DEBUG_PRINT(...)   ROS_WARN(__VA_ARGS__)
#define DEBUG_PRINT(...)   printf(__VA_ARGS__)
#endif // DEBUG_PRINT

namespace ppl_gating {

typedef people_msgs::PeoplePose PP;
typedef people_msgs::PeoplePoseList PPL;
typedef geometry_msgs::Point Pt3;

//! timeout for a track in seconds
static const double DEFAULT_TRACK_TIMEOUT = 5;
//! speed of an human in m/s. 1 m/s = 3.6 km/h
static const double DEFAULT_HUMAN_WALKING_SPEED = 10 / 3.6; // 10 km/h

/*! how long the unassigned poses must be stored (for new track creation)
 before being removed. In seconds. */
static const double DEFAULT_BLOB_UNASSIGNED_TIMEOUT = 3;
/*! half-life in seconds of an unassigned pose detection.
 Its confidence is 100% of the original after 0 half lifes,
 36% after 1 half life,  and exp(-n) after n half lifes.
  */
static const double DEFAULT_BLOB_HALF_LIFE = 2;
//! the aggregated confidence value for a blob to be converted into a new track
static const double DEFAULT_BLOB_CONFIDENCE_THRES_FOR_CONV2TRACK = 2;

////////////////////////////////////////////////////////////////////////////////

/*! determine if a given PeoplePose is inside the gate of a list of tracks
 */
inline int gate_pp2tracks(const PP & pp,
                          const PPL & tracks,
                          const double human_walking_speed = DEFAULT_HUMAN_WALKING_SPEED) {
  double best_dist = 1E10;
  int best_track_idx = assignment_utils::UNASSIGNED;
  unsigned int ntracks = tracks.poses.size();
  for (unsigned int track_idx = 0; track_idx < ntracks; ++track_idx) {
    const PP* track = &(tracks.poses[track_idx]);
    // evaluate if the distance between track center and unassociated pose
    // can be done at a human speed in the time elapsed
    double curr_time_diff = (pp.header.stamp - track->header.stamp).toSec();
    if (curr_time_diff < 0) // time soup, confused stamps => skip
      continue;
    double curr_dist = geometry_utils::distance_points3
                       (track->head_pose.position, pp.head_pose.position);
    if (best_dist <= curr_dist) // we have seen better
      continue;
    // proper metric gate
    double reachable_distance = track->std_dev + pp.std_dev +
                                human_walking_speed *  curr_time_diff;
    if (curr_dist > reachable_distance) { // too far away => skip
      DEBUG_PRINT("gate_pp2tracks(): curr_dist=%g m > reachable_distance=%g m\n",
                  curr_dist ,reachable_distance);
      continue;
    }
    best_track_idx = track_idx;
    best_dist = curr_dist;
  } // end for track_idx
  return best_track_idx;
}

////////////////////////////////////////////////////////////////////////////////

/*!
 \param new_ppl
    the measure obtained by an external method (laser, face detection...)
 \param tracks
    our current set of tracks
 \param unassociated_poses_from_new_ppl
    where will be stored the poses of new_ppl that are not associated
*/
bool gate_ppl(PPL & new_ppl,
              const PPL & tracks,
              std::vector<PP> & unassociated_poses_from_new_ppl,
              const double human_walking_speed = DEFAULT_HUMAN_WALKING_SPEED) {
  DEBUG_PRINT("gate_ppl(%i PPs, %i tracks)\n", new_ppl.poses.size(), tracks.poses.size());
  for (int pp_idx = 0; pp_idx < (int) new_ppl.poses.size(); ++pp_idx) {
    PP* pp = &(new_ppl.poses[pp_idx]);
    int best_track_idx = gate_pp2tracks(*pp, tracks, human_walking_speed);
    if (best_track_idx != assignment_utils::UNASSIGNED)
      continue;
    // PP outside of gates -> transfer to "unassociated_poses_from_new_ppl"
    unassociated_poses_from_new_ppl.push_back(*pp);
    new_ppl.poses.erase(new_ppl.poses.begin() + pp_idx);
    --pp_idx;
  } // end for pp_idx
  return true;
} // end gate_ppl();

////////////////////////////////////////////////////////////////////////////////

/*!
 \param new_ppl
    the measure obtained by an external method (laser, face detection...)
 \param tracks
    our current set of tracks
 \param costs
    the cost matrix. Is actually unchanged, but LAP, written in C, cant handle const.
 \param unassociated_poses_from_new_ppl
    where will be stored the poses of new_ppl that are not associated
 \param matches
    the pairs of correspondances new_ppl <-> tracks.
    The pairs are all valid, they are in
    ([0..new_ppl.size()], [0..tracks.size()])
*/
bool match_ppl2tracks_and_clean
(const PPL & new_ppl,
 const PPL & tracks,
 CMatrix<double> & costs,
 std::vector<PP> & unassociated_poses_from_new_ppl,
 assignment_utils::MatchList & matches) {
  DEBUG_PRINT("match_ppl2tracks_and_clean()\n");
  unsigned int nusers_detected = new_ppl.poses.size(), ntracks = tracks.poses.size();
  matches.clear();
  if (nusers_detected == 0) { // nothing to do
    return true;
  } // end if (nusers_detected == 0)
  else if (ntracks == 0) { // no linear assignment to be made
    unassociated_poses_from_new_ppl.insert(unassociated_poses_from_new_ppl.end(),
                                           new_ppl.poses.begin(),
                                           new_ppl.poses.end());
    return true;
  } // end if (ntracks == 0)
  assignment_utils::Cost best_cost;
  if (!assignment_utils::linear_assign(costs, matches, best_cost))
    return false;
  assignment_utils::clean_assign(matches);
  // find if there is any unassociated pose from new ppl
  std::vector<bool> new_ppl_is_matched(nusers_detected, false);
  for (unsigned int match_idx = 0; match_idx < matches.size(); ++match_idx)
    new_ppl_is_matched[matches[match_idx].first] = true;
  for (unsigned int new_ppl_idx = 0; new_ppl_idx < nusers_detected; ++new_ppl_idx) {
    if(!new_ppl_is_matched[new_ppl_idx])
      unassociated_poses_from_new_ppl.push_back( new_ppl.poses[new_ppl_idx]);
  } // end loop new_ppl_idx
  return true;
} // end match_ppl2tracks_and_clean();

////////////////////////////////////////////////////////////////////////////////

/*!
 * \brief update_blobs_and_create_new_tracks
 * \param time_now
 * \param unassociated_poses_from_new_ppl
 *    is cleared at the end of the function call:
 *    the poses were used for creating a blob converted into a new track
 * \param blobs
 * \param tracks
 * \param total_seen_tracks
 *    A way to create tracks with a new name for sure
 */
inline bool update_blobs_and_create_new_tracks
(const ros::Time & time_now,
 std::vector<PP> & unassociated_poses_from_new_ppl,
 PPL & blobs,
 PPL & tracks,
 unsigned int & total_seen_tracks,
 const double human_walking_speed = DEFAULT_HUMAN_WALKING_SPEED)
{
  DEBUG_PRINT("update_blobs_and_create_new_tracks(%i unassociated_poses_from_new_ppl, "
              "%i blobs, %i tracks)\n",
              unassociated_poses_from_new_ppl.size(),
              blobs.poses.size(), tracks.poses.size());

  // update blobs confidence
  for (unsigned int blob_idx = 0; blob_idx < blobs.poses.size(); ++blob_idx) {
    PP* blob = &(blobs.poses[blob_idx]);
    double initial_confidence = 1;
    if (!ppl_utils::get_attribute_readonly(*blob, "initial_confidence", initial_confidence)) {
      ppl_utils::set_attribute(*blob, "initial_confidence", 1);
      initial_confidence = 1;
    }
    double blob_age = (time_now - blob->header.stamp).toSec();
    blob->confidence = initial_confidence * exp(-blob_age / DEFAULT_BLOB_HALF_LIFE);
  } // end for blob_idx

  // check each PP from unassociated_poses_from_new_ppl and find the matching blob
  unsigned int npps = unassociated_poses_from_new_ppl.size();
  for (unsigned int pp_idx = 0; pp_idx < npps; ++pp_idx) {
    PP* pp = &(unassociated_poses_from_new_ppl[pp_idx]);
    int best_blob_idx = gate_pp2tracks(*pp, blobs, human_walking_speed);
    if (best_blob_idx == assignment_utils::UNASSIGNED)  { // no corresponding blob
      ppl_utils::set_attribute(*pp, "initial_confidence", pp->confidence);
      blobs.poses.push_back(*pp);
      continue;
    }
    // update corresponding blob
    PP* best_blob = &(blobs.poses[best_blob_idx]);
    best_blob->header = pp->header; // also update time_stamp
    best_blob->head_pose = pp->head_pose;
    best_blob->person_name = pp->person_name;
    ppl_utils::set_attribute(*best_blob, "initial_confidence",
                             best_blob->confidence + pp->confidence);
    ppl_utils::copy_attributes(*pp, *best_blob);
  } // end for pp_idx

  // check each of the blobs: if confidence big enough, create a new track
  for (int curr_blob_idx = 0; curr_blob_idx < (int) blobs.poses.size(); ++curr_blob_idx) {
    PP* blob = &(blobs.poses[curr_blob_idx]);
    if (blob->confidence <= DEFAULT_BLOB_CONFIDENCE_THRES_FOR_CONV2TRACK)
      continue;
    // transfer blob from "blobs" to "tracks"
    if (blob->person_name.empty()
        ||blob->person_name == people_msgs::PeoplePose::NO_RECOGNITION_MADE) {
      blob->person_name = std::string("track") + StringUtils::cast_to_string(total_seen_tracks);
      ++total_seen_tracks;
    }
    DEBUG_PRINT("update_blobs_and_create_new_tracks(): Creating track '%s' in %s\n",
                blob->person_name.c_str(),
                geometry_utils::printP(blob->head_pose.position).c_str());
    tracks.poses.push_back(*blob);
    blobs.poses.erase(blobs.poses.begin() + curr_blob_idx);
    --curr_blob_idx;
  } // end loop blob_idx

  unassociated_poses_from_new_ppl.clear();
  return true;
} // end update_blobs_and_create_new_tracks();

////////////////////////////////////////////////////////////////////////////////

//! remove a track if older than a given timeout
inline void remove_old_tracks(const ros::Time & time_now,
                              PPL & ppl,
                              const double & pp_timeout) {
  DEBUG_PRINT("remove_old_tracks(%i ppl)\n", ppl.poses.size());
  if (ppl.poses.empty())
    return;
  for (int pp_idx = 0; pp_idx < (int) ppl.poses.size(); ++pp_idx) {
    if ((time_now - ppl.poses[pp_idx].header.stamp).toSec() <= pp_timeout)
      continue;
    ppl.poses.erase(ppl.poses.begin() + pp_idx);
    --pp_idx;
  } // end loop pp_idx
} // end remove_old_tracks();

} // end namespace ppl_gating
#endif // PEOPLE_POSE_LIST_GATING_H
