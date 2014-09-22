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


#include <people_msgs/PeoplePoseList.h>
#include <src/ppl_utils/costs2matchlist.h>
#include "ukf_person_pose.h"

namespace ppl_gating {

typedef people_msgs::PeoplePose PP;
typedef people_msgs::PeoplePoseList PPL;

//! timeout for a track in seconds
static const double DEFAULT_TRACK_TIMEOUT = 5;
//! speed of an human in m/s. 1 m/s = 3.6 km/h
static const double MAX_HUMAN_WALKING_SPEED = 4 / 3.6; // 4 km/h

/*! how long the unassigned poses must be stored (for new track creation)
 before being removed. In seconds. */
static const double UNASSIGNED_POSE_HISTORY_TIMEOUT = 3;
/*! half-life in seconds of an unassigned pose detection.
 Its confidence is 100% of the original after 0 half lifes,
 36% after 1 half life,  and exp(-n) after n half lifes.
  */
static const double UNASSIGNED_POSE_HALF_LIFE = 2;
//! the aggregated value for a blob to be converted into a new track
static const double UNASSIGNED_POSE_BLOB_THRESH = 2;

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
bool gate_ppl
(const PPL & new_ppl,
 const PPL & tracks,
 CMatrix<double> & costs,
 std::vector<PP> & unassociated_poses_from_new_ppl,
 assignment_utils::MatchList & matches)
{
  DEBUG_PRINT("gate_ppl()\n");
  unsigned int nusers_detected = new_ppl.poses.size();
  unsigned int ntracks = tracks.poses.size();
  matches.clear();
  matches.reserve(nusers_detected);
  unassociated_poses_from_new_ppl.clear();
  unassociated_poses_from_new_ppl.reserve(nusers_detected);
  if (nusers_detected == 0) { // nothing to do
    return true;
  } // end if (nusers_detected == 0)
  else if (ntracks == 0) { // no linear assignment to be made
    unassociated_poses_from_new_ppl.insert(unassociated_poses_from_new_ppl.end(),
                                           new_ppl.poses.begin(),
                                           new_ppl.poses.end());
    return true;
  } // end if (ntracks == 0)
  if (!ppl_utils::costs_cmatrix2matchlist(costs, matches))
    return false;
  // find if there is any unassociated pose from new ppl
  std::vector<bool> new_ppl_is_matched(nusers_detected, false);
  for (unsigned int match_idx = 0; match_idx < matches.size(); ++match_idx)
    new_ppl_is_matched[matches[match_idx].first] = true;
  for (unsigned int new_ppl_idx = 0; new_ppl_idx < nusers_detected; ++new_ppl_idx) {
    if(!new_ppl_is_matched[new_ppl_idx])
      unassociated_poses_from_new_ppl.push_back( new_ppl.poses[new_ppl_idx]);
  } // end loop new_ppl_idx
  return true;
} // end gate_ppl();

////////////////////////////////////////////////////////////////////////////////

/*!
 * \brief create_new_tracks
 * \param time_now
 * \param unassociated_poses_from_new_ppl
 *    is cleared at the end of the function call:
 *    either the poses were used for creating a blob converted into a new track,
 *    or they are in unassociated_poses.
 * \param unassociated_poses
 * \param blobs
 * \param tracks
 * \param total_seen_tracks
 *    A way to create tracks with a new name for sure
 */
inline void create_new_tracks
(const ros::Time & time_now,
 std::vector<PP> & unassociated_poses_from_new_ppl,
 std::vector<PP> & unassociated_poses,
 PPL & blobs,
 PPL & tracks,
 int & total_seen_tracks)
{
  DEBUG_PRINT("create_new_tracks(%i unassociated_poses_from_new_ppl, "
              "%i unassociated_poses, %i blobs, %i tracks)\n",
              unassociated_poses_from_new_ppl.size(),
              unassociated_poses.size(), blobs.poses.size(), tracks.poses.size());

  // clean very old unassociated poses

  //  ros::Time time_now = (nusers_detected == 0 ?
  //                          ros::Time::now() :
  //                          new_ppl.poses.front().header.stamp);
  unsigned int size_before = unassociated_poses.size();
  while(true) {
    if (unassociated_poses.size() == 0)
      break;
    double front_age = (time_now - unassociated_poses.front().header.stamp).toSec();
    DEBUG_PRINT("create_new_tracks(): front_age:%g seconds\n", front_age);
    if (front_age <= UNASSIGNED_POSE_HISTORY_TIMEOUT)
      break;
    unassociated_poses.erase(unassociated_poses.begin());
  } // end while()
  if (size_before != unassociated_poses.size()) {
    DEBUG_PRINT("create_new_tracks(): Removed %i poses from unassociated_poses\n",
                size_before - unassociated_poses.size());
  }


  // for each of the new unassociated_poses_from_new_ppl ,
  // create a blob for him
  unsigned int nblobs = unassociated_poses_from_new_ppl.size(),
      nold = unassociated_poses.size();
  blobs.poses = unassociated_poses_from_new_ppl;

  // for each of the poses from unassociated_poses,
  // add it to the given blob if close enough
  std::vector<int> old2blob(nold, assignment_utils::UNASSIGNED);
  // find closest blob
  for (unsigned int old_idx = 0; old_idx < nold; ++old_idx) {
    double best_blob_dist = std::numeric_limits<double>::max(), best_time_diff = 1;
    for (unsigned int blob_idx = 0; blob_idx < nblobs; ++blob_idx) {
      // evaluate if the distance between blob center and unassociated pose
      // can be done at a human speed in the time elapsed
      double curr_blob_dist = geometry_utils::distance_points3
                              (blobs.poses[blob_idx].head_pose.position,
                               unassociated_poses[old_idx].head_pose.position);
      double curr_time_diff = (blobs.poses[blob_idx].header.stamp -
                               unassociated_poses[old_idx].header.stamp).toSec();
      if (curr_time_diff < 0) // time soup, confused stamps => skip
        continue;
      if (curr_blob_dist > MAX_HUMAN_WALKING_SPEED * curr_time_diff) // too far away => skip
        continue;
      if (best_blob_dist <= curr_blob_dist)
        continue;
      best_blob_dist = curr_blob_dist;
      old2blob[old_idx] = blob_idx;
      best_time_diff = curr_time_diff;
    } // end loop blob_idx
    // if we found a compatible blob
    if (old2blob[old_idx] == assignment_utils::UNASSIGNED)
      continue;
    // augment its confidence with a decay factor
    blobs.poses[old2blob[old_idx]].confidence +=
        unassociated_poses[old_idx].confidence *
        exp(-best_time_diff / UNASSIGNED_POSE_HALF_LIFE);
  } // end loop old_idx


  // check each of the blobs: if confidence big enough, create a new track
  std::vector<bool> blob_is_valid_track(nblobs, false);
  for (unsigned int blob_idx = 0; blob_idx < nblobs; ++blob_idx) {
    if (blobs.poses[blob_idx].confidence <= UNASSIGNED_POSE_BLOB_THRESH)
      continue;
    blob_is_valid_track[blob_idx] = true;
    // create track
    PP* blob = &(blobs.poses[blob_idx]);
    tracks.poses.push_back(*blob);
    PP* track = &(tracks.poses.back());
    if (track->person_name == people_msgs::PeoplePose::NO_RECOGNITION_MADE) {
      track->person_name = std::string("track") + StringUtils::cast_to_string(total_seen_tracks);
      ++total_seen_tracks;
     }
    DEBUG_PRINT("create_new_tracks(): Creating track '%s' in %s\n",
                track->person_name.c_str(),
                geometry_utils::printP(track->head_pose.position).c_str());
  } // end loop blob_idx

  // go backwards in unassociated_poses
  // and erase the poses used for the new blob
  for (int old_idx = unassociated_poses.size()-1; old_idx >= 0; --old_idx) {
    int blob_idx = old2blob[old_idx];
    if (blob_idx != assignment_utils::UNASSIGNED && blob_is_valid_track[blob_idx])
      unassociated_poses.erase(unassociated_poses.begin()+old_idx);
  } // end for (old_idx)
  // go backwards in blobs
  // and erase the poses used for the new blobs
  for (int blob_idx = nblobs-1; blob_idx >= 0; --blob_idx) {
    if (!blob_is_valid_track[blob_idx])
      continue;
    blobs.poses.erase(blobs.poses.begin()+blob_idx);
    unassociated_poses_from_new_ppl.erase(unassociated_poses_from_new_ppl.begin()+blob_idx);
  } // end for (blob_idx)


  // copy remaining unassociated_poses_from_new_ppl -> end of unassociated_poses
  unassociated_poses.insert(unassociated_poses.end(),
                            unassociated_poses_from_new_ppl.begin(),
                            unassociated_poses_from_new_ppl.end());

  unassociated_poses_from_new_ppl.clear();

} // end create_new_tracks();

////////////////////////////////////////////////////////////////////////////////

//! remove a track if older than a given timeout
inline void remove_old_tracks(PPL & tracks,
                              double track_timeout = DEFAULT_TRACK_TIMEOUT) {
  DEBUG_PRINT("remove_old_tracks(%i tracks)\n", tracks.poses.size());

  std::vector<PP>* poses = &(tracks.poses);
  if (poses->empty())
    return;
  ros::Time stamp = ros::Time::now();
  for (unsigned int track_idx = 0; track_idx < poses->size(); ++track_idx) {
    ros::Time pose_stamp = (*poses)[track_idx].header.stamp;
    if ((stamp - pose_stamp).toSec() <= track_timeout)
      continue;
    poses->erase(poses->begin() + track_idx);
    --track_idx;
  } // end loop track_idx
} // end remove_old_tracks();

} // end namespace ppl_gating
#endif // PEOPLE_POSE_LIST_GATING_H
