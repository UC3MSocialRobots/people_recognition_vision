/*!
  \file         conditional_particle_filter_laser.h
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

#ifndef CONDITIONNAL_PARTICLE_FILTER_LASER_H
#define CONDITIONNAL_PARTICLE_FILTER_LASER_H

#include <conditional_particle_filter.h>
#include <src/geom/geometry_utils.h>
#include <src/combinatorics/combinatorics_utils.h>

namespace conditional_particle_filter_laser {

//! a 2D point
typedef geometry_utils::FooPoint2f Pt2;

//! the sensor data "y" : the laser range.
typedef std::vector<Pt2> SensorMeasurement;

//! the pose of the human "y" : x, y, yaw
struct PeoplePose {
  float x, y, yaw;
  static const float PEOPLE_RADIUS_SQUARED = .25; // meters * meters
};

////////////////////////////////////////////////////////////////////////////////

class ConditionalParticleFilterLaser :
    public ConditionalParticleFilter<SensorMeasurement, PeoplePose> {
public:
  //////////////////////////////////////////////////////////////////////////////


  /*! the likelihood of a sensor measurement depending of the world.
      The higher, the more this \a SensorMeasurement
      corresponds to the \a WorldState.
      To be implemented by sons
      \warning the poses are given in static frame
  */
  virtual Proba sensor_measurement_model(const SensorMeasurement & z_t,
                                         const WorldState & x_t)
  {
    // \TODO it shouldn't return 0, except if the SensorMeasurement
    // really needs to be discarded
    Proba ans = 0;
    // return the number of readings corresponding to a person
    for (unsigned int t_idx = 0; t_idx < z_t.size(); ++t_idx) {
      for (unsigned int people_idx = 0; people_idx < x_t.people_poses->size();
           ++people_idx) {
        if (geometry_utils::distance_points_squared
            ((*x_t.people_poses)[people_idx], z_t[t_idx])
            < PeoplePose::PEOPLE_RADIUS_SQUARED)
          ++ans;
      } // end loop people_idx
    } // end loop idx
    return ans;
  }

  //////////////////////////////////////////////////////////////////////////////

  /*! robot motion given odometry data
      \warning the poses are given in static frame
      to implement */
  virtual RobotPose sample_robot_motion_model_law(const RobotCommandOrder & u_t,
                                                  const RobotPose & r_t_minus1,
                                                  const Timer::Time & dt_sec)
  {
    //    ROS_WARN("sample_robot_motion_model_law"
    //             "(u_t:%g, %g, r_t_minus1: %g, %g, dt_sec:%g)",
    //             u_t.linear.y, u_t.angular.z, r_t_minus1.x, r_t_minus1.y, dt_sec);
    // odometry update
    RobotPose r_t = r_t_minus1;
    odom_utils::update_pos_rot(r_t.x, r_t.y, r_t.yaw, u_t, dt_sec);
    return r_t;
  }

  //////////////////////////////////////////////////////////////////////////////

  /*! for instance brownian motion
      \warning the poses are given in static frame
      to implement */
  virtual PeoplePose sample_people_motion_model_law(const RobotCommandOrder & u_t,
                                                    const PeoplePose & y_t_minus1,
                                                    const Timer::Time & dt_sec)
  {
    // odometry update - nothing to: in the static frame, no move!
    PeoplePose y_t = y_t_minus1;

    // brownian move
    const double max_speed = .5;
    // gaussian speed
    y_t.x   += combinatorics_utils::rand_gaussian() * dt_sec * max_speed;
    y_t.y   += combinatorics_utils::rand_gaussian() * dt_sec * max_speed;
    y_t.yaw += combinatorics_utils::rand_gaussian() * dt_sec * max_speed;

    // max of .5 m/s
    //  y_t.x   += (-1 + 2 * drand48()) * dt_sec * max_speed;
    //  y_t.y   += (-1 + 2 * drand48()) * dt_sec * max_speed;
    //  y_t.yaw += (-1 + 2 * drand48()) * dt_sec * max_speed;

    return y_t;
  }

  //////////////////////////////////////////////////////////////////////////////

  virtual void init() {
    ROS_WARN("ConditionalParticleFilterLaser::init()");
    // init _R_t_minus1
    RobotPose zero_robot_pose;
    zero_robot_pose.x = 0; zero_robot_pose.y = 0; zero_robot_pose.yaw = 0;
    _R_t_minus1 = std::vector<RobotPose>(_N_r, zero_robot_pose);
    // init _Y_t_minus1
    // state of the particle filter just after initialization:
    // particles scattered all over the map
    // [robot_pose_part_idx][people_pose_part_idx][people_idx]
    std::vector<PeoplePose> v3(_M, PeoplePose());
    std::vector<std::vector<PeoplePose> > v2(_N_y, v3);
    _Y_t_minus1 = ParticleVector3(_N_r, v2);
    // init with a scattering on the  map
    float map_w = 4, map_h = 4;
    for (int i = 0; i < _N_r; ++i) {
      for (int j = 0; j < _N_y; ++j) {
        for (int m = 0; m < _M; ++m) {
          _Y_t_minus1[i][j][m].x   = (-1 + 2 * drand48()) * map_w;
          _Y_t_minus1[i][j][m].y   = (-1 + 2 * drand48()) * map_h;
          _Y_t_minus1[i][j][m].yaw = (-1 + 2 * drand48()) * M_PI;
        } // end loop m
      } // end loop j
    } // end loop i
  } // end init()

}; // end class ConditionalParticleFilterLaser

} // end namespace  conditional_particle_filter_laser

#endif // CONDITIONNAL_PARTICLE_FILTER_LASER_H
