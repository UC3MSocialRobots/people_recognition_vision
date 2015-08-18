/*!
  \file        ukf_multimodal.h
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2012/10/9

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

A node performing an Unscented Kalman Filter
on the poses of the users detected by other methods
(face detection, torso detection, etc).
This gives some more robust position detections.

\section Parameters
  - \b "~static_frame_id"
        [string] (default: "/base_link")
        The frame of the people.

  - \b "~track_timeout"
        [double, seconds] (default: ppl_gating::DEFAULT_TRACK_TIMEOUT)
        How long should we wait before cleaning a track.

  - \b "~ppl_input_topics"
        [string] (default: "ppl")
        The list of topics where to get the detections of the users
        ( of types people_msgs/PeoplePoseList ).
        Topics must be separated by ";",
        for example "/foo;/bar".

  - \b "~ppl_matcher_services"
        [string] (default: "")
        The list of services for matching detected PPL and tracks.
        Topics must be separated by ";",
        for example "/foo;/bar".
        The weights for each PPL can be specified using ":".
        For example "/foo:2;/bar:1".

  - \b "~cost_matrices_display_timeout"
        [double, seconds] (default: -1)
        If >= 0, will display the cost matrices with this time period.

  - \b "~use_gating"
        [bool] (default: true)
        Use gating?

  - \b "~human_walking_speed"
        [double, m/s] (default: ppl_gating::DEFAULT_HUMAN_WALKING_SPEED)
        The size of the gate, in meters.

\section Subscriptions
  - \b {ppl_input_topics}
        [people_msgs/PeoplePoseList]
        The different input methods for people pose lists

\section Publications
  - \b "~ppl"
        [people_msgs::PeoplePoseList]
        Results of the person tracking after applying UKF.
 */
#ifndef UKF_MULTIMODAL_H
#define UKF_MULTIMODAL_H

#include <ros_utils/pt_utils.h>
#include <ros_utils/multi_subscriber.h>
#include <time/timer.h>
// people_msgs
#include <people_msgs/MatchPPL.h>
#include "people_utils/people_pose_list_utils.h"
#include "people_utils/ppl_tf_utils.h"
#include "people_utils/ppl_attributes.h"
#include "people_utils/pplp_template.h"
// people_recognition_vision
#include "people_recognition_vision/ppl_gating.h"
// ROS
#include <ros/service_client.h>

class UkfMultiModal : public PPLPublisherTemplate {
public:
  typedef people_msgs::PeoplePose PP;
  typedef people_msgs::PeoplePoseList PPL;
  static const unsigned int QUEUE_SIZE = 1;

  UkfMultiModal() : PPLPublisherTemplate("UKF_MULTIMODAL_START", "UKF_MULTIMODAL_STOP") {
  } // end ctor

  //////////////////////////////////////////////////////////////////////////////

  void create_subscribers_and_publishers() {
    DEBUG_PRINT("UkfMultiModal::create_subscribers_and_publishers()\n");

    // get params
    _static_frame_id = "/base_link";
    _nh_private.param("static_frame_id", _static_frame_id, _static_frame_id);
    _ppl_input_topics = "ppl";
    _nh_private.param("ppl_input_topics", _ppl_input_topics, _ppl_input_topics);
    _nh_private.param("track_timeout", _track_timeout,
                      ppl_gating::DEFAULT_TRACK_TIMEOUT);
    _nh_private.param("human_walking_speed", _human_walking_speed,
                      ppl_gating::DEFAULT_HUMAN_WALKING_SPEED);
    _nh_private.param("use_gating", _use_gating, true);
    _nh_private.param("cost_matrices_display_timeout", _cost_matrices_display_timeout, -1.);
    _total_seen_tracks = 0;

    std::string services_str = "";
    _nh_private.param("ppl_matcher_services", services_str, services_str);
    ros::MultiSubscriber::split_topics_list(services_str, _matcher_services);
    unsigned int nmatchers = _matcher_services.size();

    // parse weights
    _matcher_weights.resize(nmatchers, 1.);
    for (unsigned int i = 0; i < nmatchers; ++i) {
      std::vector<std::string> words;
      StringUtils::StringSplit(_matcher_services[i], ":", &words);
      if (words.size() < 1 || words.size() > 2) {
        ROS_WARN("Error parsing service:'%s'\n", _matcher_services[i].c_str());
        continue;
      }
      // resolve names
      _matcher_services[i] = _nh_public.resolveName(words[0]);
      // parse weight
      if (words.size() == 1)
        continue;
      bool conv_success = false;
      double weight = StringUtils::cast_from_string<double>(words[1], conv_success);
      if (conv_success)
        _matcher_weights[i] = weight;
    } // end for service_idx

    // subscribe to PPLP clients
    _tf_listener = new tf::TransformListener();
    _ppl_subs = ros::MultiSubscriber::subscribe
                (_nh_public, _ppl_input_topics, QUEUE_SIZE,
                 &UkfMultiModal::ppl_cb, this);
    // blobs publisher
    _blobs_pub = _nh_public.advertise<PPL>("ukf_blobs", 1);
    // subscribe to PPLM services
    for (unsigned int topic_idx = 0; topic_idx < nmatchers; ++topic_idx)
      _matchers.push_back(_nh_public.serviceClient<people_msgs::MatchPPL>
                          (_matcher_services[topic_idx]));
    // strip services that do not exist
    sleep(2); // publishers and subscribers can need 1 sec to be ready
    _ppl_subs.strip_non_connected();
    for (int i = 0; i < _matchers.size(); ++i) {
      if (_matchers[i].exists())
        continue;
      ROS_WARN("Stripping non-existing PPLM service '%s'",
               _matchers[i].getService().c_str());
      _matcher_weights.erase(_matcher_weights.begin() + i);
      _matchers.erase(_matchers.begin() + i);
      _matcher_services.erase(_matcher_services.begin() + i);
      --i;
    }
    // check we still have PPLPs and PPLMs
    if (_ppl_subs.getNumPublishers() == 0) {
      ROS_FATAL("UkfMultiModal: you didn't specify any valid PPLP"
               "(_ppl_input_topics:=\"%s\"), "
               "please set param '~ppl_input_topics', cf doc."
                "If you use 'ukf_multimodal_lite.launch', "
                "activate at least one PPLP using "
                "<arg name=\"pplp_use_XXX\" value=\"true\"/>\n",
                _ppl_input_topics.c_str());
      ros::shutdown();
    }
    if (_matchers.empty()) {
      ROS_FATAL("UkfMultiModal: you didn't specify any valid PPLM"
               "(_ppl_matcher_services:=\"%s\"), "
               "please set param '~ppl_matcher_services', cf doc."
                "If you use 'ukf_multimodal_lite.launch', "
                "activate at least one PPLM using "
                "<arg name=\"pplm_use_XXX\" value=\"true\"/>\n",
                services_str.c_str());
      ros::shutdown();
    }

    ROS_INFO("UkfMultiModal: getting PeoplePoseList on %i topics '%s'', "
           "%i matchers on '%s' (weights:%s), "
           "track timeout of %g sec, "
           "publishing filtered PeoplePoseList on '%s', blobs on '%s'"
           "and displaying cost matrices every %g seconds."
           "use_gating:%i, human_walking_speed:%g m/s\n",
           _ppl_subs.nTopics(), _ppl_subs.getTopics().c_str(),
           _matchers.size(), StringUtils::iterable_to_string(_matcher_services).c_str(),
           StringUtils::iterable_to_string(_matcher_weights).c_str(),
           _track_timeout,
           get_ppl_topic().c_str(), _blobs_pub.getTopic().c_str(),
           _cost_matrices_display_timeout, _use_gating, _human_walking_speed);
  }

  //////////////////////////////////////////////////////////////////////////////

  void shutdown_subscribers_and_publishers() {
    DEBUG_PRINT("UkfMultiModal::shutdown_subscribers_and_publishers()\n");
    delete _tf_listener;
    _ppl_subs.shutdown();
    _blobs_pub.shutdown();
    for (unsigned int topic_idx = 0; topic_idx < _matcher_services.size(); ++topic_idx)
      _matchers[topic_idx].shutdown();
    _matchers.clear();
  }

  //////////////////////////////////////////////////////////////////////////////

  //! \return the static frame of the UKF
  inline std::string get_static_frame_id() const { return _static_frame_id; }
  //! \return the number of users
  inline unsigned int nusers() const { return _tracks.poses.size(); }
  //! \return the number of blobs after last ppl_cb()
  inline unsigned int nblobs() const { return _blobs.poses.size(); }
  //! \return the total number of publishers
  inline unsigned int nb_total_pplp() const { return _ppl_subs.nTopics(); }
  //! \return the registered number of publishers
  inline unsigned int nb_available_pplp() const { return _ppl_subs.getNumPublishers(); }
  //! \return the registered number of PPL matchers
  inline unsigned int nb_total_matchers() const { return _matchers.size(); }
  //! \return the number of available PPL matchers (is <= nb_total_matchers() )
  inline unsigned int nb_available_matchers() {
    unsigned int ans = 0;
    for (unsigned int matcher_idx = 0; matcher_idx < _matchers.size(); ++matcher_idx)
      if (_matchers[matcher_idx].exists())
        ++ans;
    return ans;
  }

  //! \return the position of each user
  template<class Pt3f>
  inline std::vector<Pt3f> get_user_positions() const {
    return ppl_utils::ppl2points<Pt3f>(_tracks);
  }

  inline std::vector<std::string> get_track_names() const {
    return ppl_utils::ppl2names(_tracks);
  }

  //! \return the affectations between the last PPL and the tracks
  inline assignment_utils::MatchList get_ppl2track_affectations() const {
    return _ppl2track_affectations;
  }

  //! \return the costs to match a new PP to an existing track
  inline const CMatrix<double> & get_avg_costs() const {
    return _avg_costs;
  }

  //////////////////////////////////////////////////////////////////////////////

  /*! the callback when a user is detected. */
  void ppl_cb(const PPL & new_ppl_bad_tf) {
    unsigned int npps = new_ppl_bad_tf.poses.size(), ntracks =  nusers();
    ros::Time now_stamp = new_ppl_bad_tf.header.stamp;
    DEBUG_PRINT("UkfMultiModal::ppl_cb(npps:%i, method:'%s')\n",
                npps, new_ppl_bad_tf.method.c_str());
    Timer timer;

    // convert to wanted frame
    people_msgs::PeoplePoseList new_ppl = new_ppl_bad_tf;
    if (!ppl_utils::convert_ppl_tf
        (new_ppl, _static_frame_id, *_tf_listener)) {
      ROS_WARN("UkfMultiModal:Could not convert new_ppl to tf '%s'\n",
             _static_frame_id.c_str());
      return;
    }

    // gate if needed
    std::vector<people_msgs::PeoplePose> unassociated_poses_from_new_ppl;
    if (_use_gating)
      ppl_gating::gate_ppl(new_ppl, _tracks, unassociated_poses_from_new_ppl,
                           _human_walking_speed);
    npps = new_ppl.poses.size();

    // compute cost matrix
    if (!_avg_costs.resize(npps, ntracks)) {
      ROS_WARN("UkfMultiModal:Could not allocate cost matrix to (%i, %i)\n",
             npps, ntracks);
      return;
    }
    _avg_costs.set_to_zero();
    unsigned int costs_size = npps * ntracks;
    unsigned int nmatchers = _matchers.size(), nmatches = 0;
    people_msgs::MatchPPLRequest req;
    people_msgs::MatchPPLResponse res;
    req.tracks = _tracks;
    req.new_ppl = new_ppl;
    // call matching services
    bool display_cost_matrix =
        (_cost_matrices_display_timeout >= 0
         && _cost_matrices_display_timer.getTimeSeconds() > _cost_matrices_display_timeout);
    for (unsigned int matcher_idx = 0; matcher_idx < nmatchers; ++matcher_idx) {
      // make the proper service call
      std::string matcher_name = _matcher_services[matcher_idx];
      res.costs.clear();
      res.new_ppl_added_attributes.clear();
      res.tracks_added_attributes.clear();
      if (!_matchers[matcher_idx].call(req, res)
          || !res.match_success) {
        ROS_WARN("UkfMultiModal::ppl_cb(): PPLM '%s' failed!\n", matcher_name.c_str());
        continue;
      }
      // add attributes
      ppl_utils::copy_attributes(res.new_ppl_added_attributes, new_ppl);
      ppl_utils::copy_attributes(res.tracks_added_attributes, _tracks);

      // mix costs of results with avg_costs
      if (res.costs.size() != costs_size) {
        ROS_WARN("UkfMultiModal::ppl_cb(): PPLM '%s' returned a cost matrix with "
               "wrong dimensions (expected %i values, got %i)\n",
               matcher_name.c_str(), costs_size, res.costs.size());
        continue;
      }
      int data_counter = 0; // copy res.costs to _avg_costs
      double matcher_weight = _matcher_weights[matcher_idx];
      for (unsigned int detec_idx = 0; detec_idx < npps; ++detec_idx) {
        for (unsigned int track_idx = 0; track_idx < ntracks; ++track_idx)
          _avg_costs[detec_idx][track_idx] += matcher_weight * res.costs[data_counter++];
      } // end for (detec_idx)
      ++nmatches;
      // display cost matrix if wanted
      if (!display_cost_matrix)
        continue;
      _cost_matrices_display_timer.reset();
      std::ostringstream out;
      for (unsigned int i = 0; i < costs_size; ++i)
        out << std::setprecision(3) << std::setw(5) << res.costs[i]
               << (i < costs_size-1 && (i+1)%ntracks == 0 ? "\n" : " \t");
      ROS_INFO("UkfMultiModal::ppl_cb(): PPLM '%s', weight:%g, cost matrix:\n%s\n",
             matcher_name.c_str(), matcher_weight, out.str().c_str());
    } // end for (matcher_idx)

    if (nmatches == 0) {
      ROS_WARN("UkfMultiModal: Could not estimate the cost matrix "
             "with any of the %i matchers ('%s')!\n",
             _matchers.size(), StringUtils::iterable_to_string(_matcher_services).c_str());
      return;
    }

    // affect the detected PPs to each user
    // store them detec_idx -> track_idx
    if (!ppl_gating::match_ppl2tracks_and_clean
        (new_ppl, _tracks, _avg_costs,
         unassociated_poses_from_new_ppl, _ppl2track_affectations))
      return;
    DEBUG_PRINT("after match_ppl2tracks_and_clean(), "
                "affectations:'%s', unassociated_poses_from_new_ppl:size:%i\n",
                assignment_utils::assignment_list_to_string(_ppl2track_affectations).c_str(),
                unassociated_poses_from_new_ppl.size());

    // compute UKF
    for (unsigned int affec_idx = 0; affec_idx < _ppl2track_affectations.size(); ++affec_idx) {
      int detec_idx = _ppl2track_affectations[affec_idx].first;
      int track_idx = _ppl2track_affectations[affec_idx].second;
      PP* track = &(_tracks.poses.at(track_idx));
      PP* detec = &(new_ppl.poses.at(detec_idx));
      //DEBUG_PRINT("UkfMultiModal: detec_idx:%i, track_idx:%i, "
      //             "list->poses: size %i, tracks: size %i\n",
      //             detec_idx, track_idx, new_ppl.poses.size(), tracks.size());
      geometry_msgs::Point track_pos = track->head_pose.position;
      double track_orien = 0, track_speed = 0;
      if (!ppl_utils::get_attribute_readonly(*track, "ukf_orien", track_orien)) {
        ppl_utils::set_attribute(*track, "ukf_orien", 0);
        track_orien = 0;
      }
      if (!ppl_utils::get_attribute_readonly(*track, "ukf_speed", track_speed)) {
        ppl_utils::set_attribute(*track, "ukf_speed", 0);
        track_speed = 0;
      }
      // call UKF
      double delta_t_sec = (detec->header.stamp - track->header.stamp).toSec();
      UkfPersonPose ukf_pp(track_pos, track_orien, track_speed);
      geometry_msgs::Point detec_pos = detec->head_pose.position;
      // maggiePrint("detec_pos:'%s'", geometry_utils::printP(detec_pos).c_str());
      ukf_pp.measurement_update(detec_pos, delta_t_sec);

      // refresh PP
      track->header = detec->header;
      ukf_pp.get_state(track->head_pose.position, track_orien, track_speed);
      ppl_utils::set_attribute(*track, "ukf_orien", track_orien);
      ppl_utils::set_attribute(*track, "ukf_speed", track_speed);
      if (!detec->person_name.empty()
          && detec->person_name != people_msgs::PeoplePose::NO_RECOGNITION_MADE
          && detec->person_name != people_msgs::PeoplePose::RECOGNITION_FAILED)
        track->person_name = detec->person_name;
      track->std_dev = ukf_pp.get_std_dev();
      track->confidence = detec->confidence; // TODO improve that
      if (detec->rgb.width > 0 && detec->rgb.height > 0) { // copy images
        track->rgb = detec->rgb;
        track->depth = detec->depth;
        track->user = detec->user;
        track->images_offsetx = detec->images_offsetx;
        track->images_offsety = detec->images_offsety;
      }
      // copy all other attributes
      ppl_utils::copy_attributes(*detec, *track);
    } // end for affec_idx

    ppl_gating::remove_old_tracks(now_stamp, _tracks, _track_timeout);
    ppl_gating::remove_old_tracks(now_stamp, _blobs, ppl_gating::DEFAULT_BLOB_UNASSIGNED_TIMEOUT);
    ppl_gating::update_blobs_and_create_new_tracks
        (now_stamp, unassociated_poses_from_new_ppl, _blobs, _tracks,
         _total_seen_tracks, _human_walking_speed);

    // print result
    DEBUG_PRINT("UkfMultiModal:tracks:%s\n", ppl_utils::ppl2string(_tracks).c_str());
    DEBUG_PRINT("UkfMultiModal:blobs:%s\n", ppl_utils::ppl2string(_blobs).c_str());

    // publish message
    _tracks.header.stamp = ros::Time::now();
    _tracks.header.frame_id = _static_frame_id;
    _tracks.method = ros::this_node::getName();
    publish_PPL(_tracks);
    _blobs.header = _tracks.header;
    _blobs.method = "ukf_multimodal_blobs";
    _blobs_pub.publish(_blobs);
    DEBUG_PRINT("UkfMultiModal: Time for ppl_cb(): %g ms\n", timer.getTimeMilliseconds());
  } // ppl_cb();

  //////////////////////////////////////////////////////////////////////////////

private:
  std::string _static_frame_id;
  tf::TransformListener* _tf_listener;
  PPL _tracks;
  unsigned int _total_seen_tracks;

  // gating stuff
  bool _use_gating;
  assignment_utils::MatchList _ppl2track_affectations;
  PPL _blobs;
  double _track_timeout, _human_walking_speed;

  // input PPls
  std::string _ppl_input_topics;
  ros::MultiSubscriber _ppl_subs;

  // matchers
  CMatrix<double> _avg_costs;
  std::vector<std::string> _matcher_services;
  std::vector<double> _matcher_weights;
  std::vector<ros::ServiceClient> _matchers;

  // outputs
  ros::Publisher _blobs_pub;
  double _cost_matrices_display_timeout;
  Timer _cost_matrices_display_timer;
}; // end class UkfMultiModal

#endif // UKF_MULTIMODAL_H
