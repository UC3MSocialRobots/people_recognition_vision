/*!
  \file         conditional_particle_filter_laser_ros.h
  \author       Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date         2012/1

  ______________________________________________________________________________

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
  ______________________________________________________________________________

  */

#ifndef CONDITIONAL_PARTICLE_FILTER_LASER_ROS_H
#define CONDITIONAL_PARTICLE_FILTER_LASER_ROS_H

// ROS
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/LaserScan.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_listener.h>
#include "ros_utils/laser_utils.h"
#include "color/color_utils.h"
// conditional_particle_filter
#include "conditional_particle_filter_laser.h"

namespace conditional_particle_filter_laser {

////////////////////////////////////////////////////////////////////////////////

class ConditionalParticleFilterLaserRos : public ConditionalParticleFilterLaser {
public:
  typedef geometry_msgs::TwistStamped               VelCmd;
  typedef geometry_msgs::TwistStamped::ConstPtr     VelCmdConstPtr;
  typedef sensor_msgs::LaserScan                    LaserMsg;
  typedef sensor_msgs::LaserScan::ConstPtr          LaserMsgConstPtr;
  typedef message_filters::sync_policies::ApproximateTime<LaserMsg, VelCmd>
  DataSyncPolicy;

private:
  ros::NodeHandle _nh_public;
  message_filters::Subscriber<LaserMsg> _laser_subscriber;
  message_filters::Subscriber<VelCmd> _com_vel_subscriber;

public:
  //////////////////////////////////////////////////////////////////////////////

  //! ctor
  ConditionalParticleFilterLaserRos() :
    _laser_subscriber(_nh_public, "scan_filtered", 1),
    _com_vel_subscriber(_nh_public, "cmd_vel_stamped", 1)
  {

    ros::NodeHandle nh_private("~");
    // get static_frame
    nh_private.param("static_frame", static_frame, std::string("/odom"));

    // prepair _marker
    _marker.header.frame_id = static_frame;
    _marker.type = visualization_msgs::Marker::SPHERE_LIST;
    _marker.action = visualization_msgs::Marker::ADD;
    _marker.ns = "conditional_particle_filter";
    _marker.lifetime = ros::Duration(1);
    _marker.scale.x = .2;
    _marker.scale.y = .2;
    _marker.scale.z = .2;
    _marker.points.reserve(_N_r * _N_y * _M + _N_r);
    _marker.colors.reserve(_N_r * _N_y * _M + _N_r);
    _marker.color.r = 1; _marker.color.g = 1; _marker.color.b = 1; _marker.color.a = 1;
    for (int i = 0; i < _N_r; ++i) {
      for (int j = 0; j < _N_y; ++j) {
        for (int m = 0; m < _M; ++m) {
          // put a color depending on the person index
          _marker.colors.push_back(std_msgs::ColorRGBA());
          _marker.colors.back().a = 1;
          color_utils::indexed_color_norm
              (_marker.colors.back().r,
               _marker.colors.back().g,
               _marker.colors.back().b,
               m);
        } // end loop m
      } // end loop j
    } // end loop i

    // add white for the robot pose particles
    for (int i = 0; i < _N_r; ++i) {
      _marker.colors.push_back(std_msgs::ColorRGBA());
      _marker.colors.back().r = _marker.colors.back().g = 1;
      _marker.colors.back().b = _marker.colors.back().a = 1;
    } // end loop i

    sleep(1);

    // make subscribers
    // ApproximateTime synchronizer
    // ApproximateTime takes a queue size as its constructor argument
    //policy.setMaxIntervalDuration (ros::Duration(30.f / 1000)); // max package of 30ms
    //    message_filters::Synchronizer<DataSyncPolicy> sync
    //        (DataSyncPolicy(1), _laser_subscriber, _com_vel_subscriber);
    //    // register the call back
    //    message_filters::Connection conn = sync.registerCallback
    //        (boost::bind
    //         (&ConditionalParticleFilterLaserRos::data_callback, this, _1, _2));

    laser_sub = _nh_public.subscribe
        ("scan_filtered", 1, &ConditionalParticleFilterLaserRos::laser_cb, this);
    //    vel_sub = _nh_public.subscribe
    //        ("cmd_vel_stamped", 1, &ConditionalParticleFilterLaserRos::vel_cb, this);

    _marker_pub = nh_private.advertise<visualization_msgs::Marker>
        ("particle_marker", 1);



  } // end ctor

  //////////////////////////////////////////////////////////////////////////////

  void laser_cb(const LaserMsgConstPtr & laser_msg) {
    //ROS_INFO("laser_cb()");
    // wait for a speed order
    VelCmdConstPtr cmd_vel_msg = ros::topic::waitForMessage<VelCmd>
        ("cmd_vel_stamped", _nh_public, ros::Duration(1));
    if (cmd_vel_msg)
      data_callback(laser_msg, cmd_vel_msg);
  }

  //////////////////////////////////////////////////////////////////////////////

  //! this function is called each time an image is received
  void data_callback(const LaserMsgConstPtr & laser_msg,
                     const VelCmdConstPtr & cmd_vel_msg) {
    ROS_INFO_THROTTLE(1, "data_callback()");
    // get the robot order
    RobotCommandOrder u_t;
    pt_utils::copy3(cmd_vel_msg->twist.linear, u_t.linear);
    pt_utils::copy3(cmd_vel_msg->twist.angular, u_t.angular);

    // build the laser data
    SensorMeasurement z_t;
    laser_utils::convert_sensor_data_to_xy(*laser_msg, z_t);
    // convert to static_frame
    geometry_msgs::PointStamped pt_stamped_in, pt_stamped_out;
    pt_stamped_in.point.z = 0;
    pt_stamped_in.header = laser_msg->header;
    for (unsigned int pt_idx = 0; pt_idx < z_t.size(); ++pt_idx) {
      pt_utils::copy2(z_t[pt_idx], pt_stamped_in.point);
      _tf_listener.transformPoint(static_frame,  ros::Time(0),
                                  pt_stamped_in, static_frame, pt_stamped_out);
      pt_utils::copy2(pt_stamped_out.point, z_t[pt_idx]);
    } // end loop pt_idx


    // call the algorithm
    algo_with_storage(u_t, z_t);

    // make a marker message
    _marker.header.stamp = laser_msg->header.stamp;
    _marker.points.clear();
    for (int i = 0; i < _N_r; ++i) {
      for (int j = 0; j < _N_y; ++j) {
        for (int m = 0; m < _M; ++m) {
          _marker.points.push_back(geometry_msgs::Point());
          // [robot_pose_part_idx][people_pose_part_idx][people_idx]
          _marker.points.back().x = _Y_t[i][j][m].x;
          _marker.points.back().y = _Y_t[i][j][m].y;
        } // end loop m
      } // end loop j
    } // end loop i

    // add robot pose particles
    for (int i = 0; i < _N_r; ++i) {
      _marker.points.push_back(geometry_msgs::Point());
      _marker.points.back().x = _R_t[i].x;
      _marker.points.back().y = _R_t[i].y;
    } // end loop i

    //    ROS_WARN("Points of size %i, colors of size %i",
    //             _marker.points.size(), _marker.colors.size());
    _marker_pub.publish(_marker);
  } // end data_callback();

  //////////////////////////////////////////////////////////////////////////////

  virtual void init() {
    ROS_WARN("ConditionalParticleFilterLaserRos::init()");
    ConditionalParticleFilterLaser::init();
    // set the zero pose
    geometry_msgs::PoseStamped pose_stamped_in, pose_stamped_out;
    pose_stamped_in.header.frame_id = "/base_link";
    pose_stamped_in.pose.orientation = tf::createQuaternionMsgFromYaw(0);
    _tf_listener.transformPose(static_frame,  ros::Time(0),
                               pose_stamped_in, static_frame, pose_stamped_out);
    RobotPose zero_robot_pose;
    zero_robot_pose.yaw = tf::getYaw(pose_stamped_out.pose.orientation);
    pt_utils::copy2(pose_stamped_out.pose.position, zero_robot_pose);
    _R_t_minus1 = std::vector<RobotPose>(_N_r, zero_robot_pose);
  } // end init()


private:
  visualization_msgs::Marker _marker;
  ros::Publisher _marker_pub;
  std::string static_frame;
  ros::Subscriber laser_sub;
  tf::TransformListener _tf_listener;

}; // end class ConditionalParticleFilterLaserRos

} // end namespace  conditional_particle_filter_laser

#endif // CONDITIONAL_PARTICLE_FILTER_LASER_ROS_H
