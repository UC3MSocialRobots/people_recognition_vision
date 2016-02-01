/*!
  \file        pplm_benchmarker.cpp
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2016/1/31

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
#include "vision_utils/images2ppl.h"
#include "vision_utils/rlpd2imgs.h"
#include "vision_utils/utils/assignment_utils.h"
#include "vision_utils/utils/map_utils.h"
#include "vision_utils/utils/string_casts_stl.h"
#include "vision_utils/utils/timer.h"
// people_msgs
#include <people_msgs/MatchPPL.h>
#include <ros/ros.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "pplm_benchmarker");
  // get input files
  if (argc < 2) {
    ROS_WARN("Synopsis: %s [RGB files]", argv[0]);
    return -1;
  }
  std::ostringstream files;
  for (int argi = 1; argi < argc; ++argi) // 0 is the name of the exe
    files << argv[argi]<< ";";
  RLPD2Imgs reader;
  bool repeat = false;
  if (!reader.from_file(files.str(), repeat)) {
    ROS_ERROR("Could not parse files '%s'", files.str().c_str());
    return -1;
  }

  // get params
  ros::NodeHandle nh_public, nh_private("~");
  bool display = false;
  nh_private.param("display", display, display);
  std::string names_str = "";
  nh_private.param("names", names_str, names_str);
  if (names_str.empty()) {
    ROS_ERROR("Parameter 'names' must be defined, for instance 'joe;jane'");
    return -1;
  }
  std::map<std::string, int> name2idx;
  std::vector<std::string> names;
  string_utils::StringSplit(names_str, ";", &names);
  unsigned int nnames = names.size();
  for (int i = 0; i < nnames; ++i)
    name2idx.insert(std::pair<std::string, unsigned int>(names[i], name2idx.size()));

  // service
  ros::ServiceClient matcher = nh_public.serviceClient<people_msgs::MatchPPL>("match_ppl");

  // start the loop
  cv::Mat1d confusion_matrix(nnames, nnames);
  confusion_matrix.setTo(0);
  ppl_utils::Images2PPL ground_truth_ppl_conv;
  people_msgs::MatchPPLRequest req;
  people_msgs::MatchPPLResponse res;
  std_msgs::Header curr_header;
  ROS_INFO("pplm_benchmarker: service '%s', names: '%s'",
           matcher.getService().c_str(), string_utils::map_to_string(name2idx).c_str());

  curr_header.frame_id = "openni_rgb_optical_frame";
  curr_header.stamp = ros::Time::now();
  while (ros::ok()) {
    Timer timer;
    if (!reader.go_to_next_frame()) {
      ROS_WARN("pplm_benchmarker: couldn't go_to_next_frame()!");
      break;
    }
    ROS_INFO_THROTTLE(10, "Time for go_to_next_frame(): %g ms.", timer.getTimeMilliseconds());

    // transform to PPL
    curr_header.stamp += ros::Duration(.2); // 5 Hz
    if (!ground_truth_ppl_conv.convert(reader.get_bgr(),
                                       reader.get_depth(),
                                       reader.get_ground_truth_user(),
                                       NULL,
                                       &curr_header)) {
      ROS_WARN("ground_truth_ppl_conv.convert() failed!");
      continue;
    }

    // call "MatchPPL" service
    req.tracks = req.new_ppl;
    req.new_ppl = ground_truth_ppl_conv.get_ppl();
    unsigned int ntracks = req.tracks.poses.size(), nppl = req.new_ppl.poses.size();
    if (nppl < 2) {
      ROS_WARN_THROTTLE(5, "Only %i users, no recognition to be made!", nppl);
      continue;
    }

    if (!matcher.call(req, res) || !res.match_success) {
      ROS_WARN("Service '%s' failed!", matcher.getService().c_str());
      continue;
    }
    // mix costs of results with avg_costs
    unsigned int costs_size = nppl * ntracks;
    if (res.costs.size() != costs_size) {
      ROS_WARN("pplm_benchmarker::ppl_cb(): PPLM '%s' returned a cost matrix with "
             "wrong dimensions (expected %i values, got %i)",
             matcher.getService().c_str(), costs_size, res.costs.size());
      continue;
    }

    // compute assignment
    assignment_utils::Cost best_cost;
    assignment_utils::MatchList ppl2track_affectations;
    if (!assignment_utils::linear_assign_from_cost_vec
        (res.costs, nppl, ntracks, ppl2track_affectations, best_cost)) {
      ROS_WARN("linear_assign_from_cost_vec() failed!");
      continue;
    }
    std::cout << assignment_utils::assignment_list_to_string(ppl2track_affectations) << std::endl;

    // check if the recognition is correct
    unsigned int nassigns = ppl2track_affectations.size();
    for (int i = 0; i < nassigns; ++i) {
      int ppli = ppl2track_affectations[i].first, tracki  = ppl2track_affectations[i].second;
      if (ppli == assignment_utils::UNASSIGNED || tracki == assignment_utils::UNASSIGNED) {
        ROS_WARN_THROTTLE(5, "Uncomplete assignment! '%s'",
                 assignment_utils::assignment_list_to_string(ppl2track_affectations).c_str());
        continue;
      }
      std::string ppl_name,track_name;
      if (!ppl_utils::get_attribute_readonly(req.new_ppl.poses[ppli], "user_multimap_name", ppl_name)
          || !ppl_utils::get_attribute_readonly(req.tracks.poses[tracki], "user_multimap_name", track_name)) {
        ROS_WARN("Couldn't get names!");
        continue;
      }
      //printf("Match %i: PPL '%s' <-> track '%s'\n", i, ppl_name.c_str(), track_name.c_str());
      // store in confusion matrix
      if (!name2idx.count(ppl_name) || !name2idx.count(track_name)) {
        ROS_WARN("Unknown PPL '%s' or track '%s'", ppl_name.c_str(), track_name.c_str());
        continue;
      }
      unsigned int col = name2idx[ppl_name] , row = name2idx[track_name];
      //printf("('%s', '%s') -> (%i, %i)\n", track_name.c_str(), ppl_name.c_str(), col, row);
      ++(confusion_matrix.at<double>(row, col)); // img.at<uchar>(y, x)
    } // end for i
    // print confusion matrix
    std::cout << confusion_matrix << std::endl;

    // display
    if (display)
      reader.display();
    ros::spinOnce();
  } // end while(ros::ok())
  // normalize confusion matrix
  // http://www.marcovanetti.com/pages/cfmatrix/?noc=3
  // https://en.wikipedia.org/wiki/Confusion_matrix
  // rows: tracks (true labels), cols: detections (matching result)
  // -> we need to normalize by row
  for (int i = 0; i < nnames; ++i)
    cv::normalize(confusion_matrix.row(i), confusion_matrix.row(i));
  std::cout << confusion_matrix << std::endl;
  return 0;
}
