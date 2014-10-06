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
  - \b "static_frame_id"
        [string] (default: "/base_link")
        The frame of the people.

  - \b "track_timeout"
        [double, seconds] (default: ppl_gating::DEFAULT_TRACK_TIMEOUT)
        How long should we wait before cleaning a track.

  - \b "ppl_input_topics"
        [string] (default: "ppl")
        The list of topics where to get the detections of the users
        ( of types people_msgs/PeoplePoseList ).
        Topics must be separated by ";",
        for example "/foo;/bar".

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
#include <ppl_utils/ppl_tf_utils.h>
#include <ppl_utils/ppl_attributes.h>
#include <templates/pplp_template.h>
// people_recognition_vision
#include "ppl_gating.h"

class UkfMultiModal : public PPLPublisherTemplate {
public:
  typedef people_msgs::PeoplePose PP;
  typedef people_msgs::PeoplePoseList PPL;
  static const unsigned int QUEUE_SIZE = 1;

  UkfMultiModal() : PPLPublisherTemplate("UKF_MULTIMODAL_START", "UKF_MULTIMODAL_STOP") {
    // get params
    _static_frame_id = "/base_link";
    _nh_private.param("static_frame_id", _static_frame_id, _static_frame_id);
    _ppl_input_topics = "ppl";
    _nh_private.param("ppl_input_topics", _ppl_input_topics, _ppl_input_topics);
    _nh_private.param("track_timeout", _track_timeout, ppl_gating::DEFAULT_TRACK_TIMEOUT);

    std::string services = "";
    _nh_private.param("ppl_matcher_services", services, services);
    ros::MultiSubscriber::split_topics_list(services, _matcher_services);
    if (_matcher_services.size() == 0) {
      printf("UkfMultiModal: you didn't specify any matcher, "
             "please set param '~ppl_matcher_services' and see doc.\n");
      //ros::shutdown();
    }
    for (unsigned int i = 0; i < _matcher_services.size(); ++i)
      _matcher_services[i] = _nh_public.resolveName(_matcher_services[i]);
    _total_seen_tracks = 0;
  } // end ctor

  //////////////////////////////////////////////////////////////////////////////

  void create_subscribers_and_publishers() {
    DEBUG_PRINT("UkfMultiModal::create_subscribers_and_publishers()\n");
    // PPL subscribers
    _tf_listener = new tf::TransformListener();
    _ppl_subs = ros::MultiSubscriber::subscribe
        (_nh_public, _ppl_input_topics, QUEUE_SIZE,
         &UkfMultiModal::ppl_cb, this);
    // matchers
    for (unsigned int topic_idx = 0; topic_idx < _matcher_services.size(); ++topic_idx)
      _matchers.push_back(_nh_public.serviceClient<people_msgs::MatchPPL>
                          (_matcher_services[topic_idx]));
    _matcher_services_concat = StringUtils::iterable_to_string(_matcher_services);
    _blobs_pub = _nh_public.advertise<PPL>("ukf_blobs", 1);

    printf("UkfMultiModal: getting PeoplePoseList on %i topics (%s), "
           "%i matchers on '%s', "
           "track timeout of %g sec, "
           "publishing filtered PeoplePoseList on '%s', blobs on '%s'\n",
           _ppl_subs.nTopics(), _ppl_subs.getTopics().c_str(),
           _matchers.size(), _matcher_services_concat.c_str(),
           _track_timeout,
           get_ppl_topic().c_str(), _blobs_pub.getTopic().c_str());
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
  //! \return the registered number of publishers
  inline unsigned int nb_ppl_input_topics() const { return _ppl_subs.getNumPublishers(); }
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
    std::vector<Pt3f> ans(nusers());
    for (unsigned int user_idx = 0; user_idx < nusers(); ++user_idx)
      pt_utils::copy3(_tracks.poses[user_idx].head_pose.position,
                      ans[user_idx]);
    return ans;
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
    unsigned int nusers_detected = new_ppl_bad_tf.poses.size(), ntracks =  nusers();
    ros::Time now_stamp = new_ppl_bad_tf.header.stamp;
    DEBUG_PRINT("UkfMultiModal::ppl_cb(nusers_detected:%i, method:'%s')\n",
                nusers_detected, new_ppl_bad_tf.method.c_str());
    Timer timer;

    // convert to wanted frame
    people_msgs::PeoplePoseList new_ppl = new_ppl_bad_tf;
    if (!ppl_utils::convert_ppl_tf
        (new_ppl, _static_frame_id, *_tf_listener)) {
      printf("UkfMultiModal:Could not convert new_ppl to tf '%s'\n",
             _static_frame_id.c_str());
      return;
    }


    // compute cost matrix
    if (!_avg_costs.resize(nusers_detected, ntracks)) {
      printf("UkfMultiModal:Could not allocate cost matrix to (%i, %i)\n",
             nusers_detected, ntracks);
      return;
    }

    _avg_costs.set_to_zero();
    unsigned int costs_size = nusers_detected * ntracks;
    unsigned int nmatchers = _matchers.size(), nmatches = 0;
    people_msgs::MatchPPLRequest req;
    people_msgs::MatchPPLResponse res;
    req.tracks = _tracks;
    req.new_ppl = new_ppl;
    // call matching services
    for (unsigned int matcher_idx = 0; matcher_idx < nmatchers; ++matcher_idx) {
      if (!_matchers[matcher_idx].call(req, res)
          || !res.match_success) {
        ROS_WARN_ONCE("gate_ppl(): client '%s' failed!\n",
                      _matchers[matcher_idx].getService().c_str());
        continue;
      }
      // mix costs of results with avg_costs
      if (res.costs.size() != costs_size) {
        printf("gate_ppl(): client '%s' returned a cost matrix with "
               "wrong dimensions (expected %i values, got %i)\n",
               _matchers[matcher_idx].getService().c_str(),
               costs_size, res.costs.size());
        continue;
      }
      int data_counter = 0;
      for (unsigned int detec_idx = 0; detec_idx < nusers_detected; ++detec_idx) {
        for (unsigned int track_idx = 0; track_idx < ntracks; ++track_idx)
          _avg_costs[detec_idx][track_idx] += res.costs[data_counter++];
      } // end for (detec_idx)
      ++nmatches;
    } // end for (matcher_idx)

    if (nmatches == 0) {
      ROS_WARN_ONCE("UkfMultiModal: Could not estimate the cost matrix "
                    "with any of the %i matchers ('%s')!",
                    _matchers.size(),
                    StringUtils::iterable_to_string(_matcher_services).c_str());
      return;
    }
    // normalize avg_costs

    double nmatches_inv = 1. / nmatchers;
    for (unsigned int detec_idx = 0; detec_idx < nusers_detected; ++detec_idx) {
      for (unsigned int track_idx = 0; track_idx < ntracks; ++track_idx)
        _avg_costs[detec_idx][track_idx] *= nmatches_inv;
    } // end for (detec_idx)

    // affect the detected PPs to each user
    // store them detec_idx -> track_idx

    std::vector<people_msgs::PeoplePose> unassociated_poses_from_new_ppl;
    if (!ppl_gating::gate_ppl
        (new_ppl, _tracks, _avg_costs,
         unassociated_poses_from_new_ppl, _ppl2track_affectations))
      return;
    DEBUG_PRINT("after gate_ppl(), "
                "affectations:'%s', unassociated_poses_from_new_ppl:size:%i",
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
      ppl_utils::get_attribute(*track, "ukf_orien", track_orien, true);
      ppl_utils::get_attribute(*track, "ukf_speed", track_speed, true);

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
      track->std_dev = ukf_pp.get_std_dev();
      track->confidence = detec->confidence; // TODO improve that
      if (detec->rgb.width > 0 && detec->rgb.height > 0) { // copy images
        track->rgb = detec->rgb;
        track->depth = detec->depth;
        track->user = detec->user;
        track->images_offsetx = detec->images_offsetx;
        track->images_offsety = detec->images_offsety;
      }
    } // end for affec_idx

    ppl_gating::remove_old_tracks(_tracks, _track_timeout);
    ppl_gating::create_new_tracks
        (now_stamp, unassociated_poses_from_new_ppl, _unassociated_poses,
         _blobs, _tracks, _total_seen_tracks);

    // print result
    DEBUG_PRINT("UkfMultiModal: %i users, %i blobs, %i unassociated poses\n",
                nusers(), nblobs(), _unassociated_poses.size());
    for (unsigned int track_idx = 0; track_idx < nusers(); ++track_idx) {
      double track_orien = 0, track_speed = 0;
      ppl_utils::get_attribute(_tracks.poses[track_idx], "ukf_orien", track_orien);
      ppl_utils::get_attribute(_tracks.poses[track_idx], "ukf_speed", track_speed);
      DEBUG_PRINT("UkfMultiModal: track user #%i pt:%s, orien:%g, speed:%g\n", track_idx,
                  geometry_utils::printP(_tracks.poses[track_idx].head_pose.position).c_str(),
                  track_orien, track_speed);
    } // end loop track_idx
    for (unsigned int blob_idx = 0; blob_idx < nblobs(); ++blob_idx) {
      DEBUG_PRINT("UkfMultiModal: Blob #%i pos:%s, confidence:%g\n", blob_idx,
                  geometry_utils::printP(_blobs.poses[blob_idx].head_pose.position).c_str(),
                  _blobs.poses[blob_idx].confidence);
    } // end for (blob_idx)


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
  std::vector<people_msgs::PeoplePose> _unassociated_poses;
  int _total_seen_tracks;

  // gating stuff
  assignment_utils::MatchList _ppl2track_affectations;
  PPL _blobs;
  double _track_timeout;

  // input PPls
  std::string _ppl_input_topics;
  ros::MultiSubscriber _ppl_subs;

  // matchers
  CMatrix<double> _avg_costs;
  std::vector<std::string> _matcher_services;
  std::string _matcher_services_concat;
  std::vector<ros::ServiceClient> _matchers;

  // outputs
  ros::Publisher _blobs_pub;
}; // end class UkfMultiModal

#endif // UKF_MULTIMODAL_H
