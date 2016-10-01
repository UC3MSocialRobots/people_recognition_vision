/*!
  \file         conditional_particle_filter.h
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

  An implementation of
  "Conditional particle filters for simultaneous mobile robot
  localization and people-tracking"
  by Montemerlo, M.; Thrun, S.; Whittaker, W.;
  in 2002

  http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1013439&tag=1

  */

#ifndef CONDITIONALPARTICLEFILTER_H
#define CONDITIONALPARTICLEFILTER_H

// people_msgs_rl
#include "vision_utils/utils/select_index_wth_probas.h"
#include "vision_utils/utils/odom_utils.h"
#include "vision_utils/utils/timer.h"

namespace conditional_particle_filter_laser {

//! a probability
typedef float Proba;

//! the pose of the robot "r" : x, y, yaw
struct RobotPose {
  float x, y, yaw;
};

typedef odom_utils::FooRobotCommandOrder RobotCommandOrder;

//////////////////////////////////////////////////////////////////////////////

template<class SensorMeasurement, class PeoplePose>
class ConditionalParticleFilter {
public:
  /*! the world state "x" : contains the PeoplePose "y1..M",
    and also the RobotPose "r" */
  struct WorldState {
    RobotPose* robot_pose;
    std::vector<PeoplePose>* people_poses;
    //! ctor
    WorldState(RobotPose* _robot_pose, std::vector<PeoplePose>* _people_poses) :
      robot_pose(_robot_pose), people_poses(_people_poses) {}
  };

  //! all the particles [robot_pose_part_idx][people_pose_part_idx][people_idx]
  typedef std::vector< std::vector< std::vector<PeoplePose > > > ParticleVector3;


  //! number of particles for RobotPose estimation
  static const int _N_r = 1;
  //! number of particles for PeoplePose estimation
  static const int _N_y = 10;
  //! number of tracked persons
  static const int _M = 2;

  ////////////////////////////////////////////////////////////////////////////////

  //! ctor
  ConditionalParticleFilter() {
    _timer.reset();
  }

  //////////////////////////////////////////////////////////////////////////////

  /*! the likelihood of a sensor measurement depending of the world.
      The higher, the more this \a SensorMeasurement
      corresponds to the \a WorldState.
      To be implemented by sons
      \warning the poses are given in static frame
  */
  virtual Proba sensor_measurement_model(const SensorMeasurement & z_t,
                                         const WorldState & x_t)
  = 0;

  //////////////////////////////////////////////////////////////////////////////

  /*! robot motion given odometry data
      To be implemented by sons
      \warning the poses are given in static frame
  */
  virtual RobotPose sample_robot_motion_model_law(const RobotCommandOrder & u_t,
                                                  const RobotPose & r_t_minus1,
                                                  const Timer::Time & dt_sec)
  = 0;

  //////////////////////////////////////////////////////////////////////////////

  /*! for instance brownian motion
      To be implemented by sons
      \warning the poses are given in static frame
  */
  virtual PeoplePose sample_people_motion_model_law(const RobotCommandOrder & u_t,
                                                    const PeoplePose & y_t_minus1,
                                                    const Timer::Time & dt_sec)
  = 0;

  //////////////////////////////////////////////////////////////////////////////


  /*!
    \arg R_t : set of estimated RobotPose
    \arg Y_t : for each estimated RobotPose, set of estimated PeoplePose
                  for each person
    */
  inline void algo(const RobotCommandOrder & u_t,
                   const SensorMeasurement & z_t) {
    //ROS_WARN("algo()");
    //! get the time elapsed
    Timer::Time time_elapsed = _timer.getTimeSeconds();
    _timer.reset();
    // copy all previous values
    _R_t = _R_t_minus1;
    _Y_t = _Y_t_minus1;

    //! contains the probability of the persons distribution for particle (i,j)
    Proba w2[_N_r][_N_y], w1[_N_r];

    /*
     * sample position of people
     */
    // i: pose index
    for (int i = 0; i < _N_r; ++i) {
      //ROS_WARN("Step 1:i=%i", i);
      // sample the new position of the robot
      const RobotPose & r_t_minus1_i = _R_t_minus1[i];
      _R_t[i] = sample_robot_motion_model_law(u_t, r_t_minus1_i, time_elapsed);


      // j: particle index for a given people
      for (int j = 0; j < _N_y; ++j) {
        // m: people index
        for (int m = 0; m < _M; ++m) {
          // sample the new position of the person "y_m_t_ij"
          // [robot_pose_part_idx][people_pose_part_idx][people_idx]
          const PeoplePose & y_m_t_minus1_ij = _Y_t_minus1[i][j][m];
          _Y_t[i][j][m]= sample_people_motion_model_law
              (u_t, y_m_t_minus1_ij, time_elapsed);
        } // end loop m
        // now, for each person, its particle position
        // have been sampled for this RobotPose

        WorldState x_ij_t(&_R_t[i], &_Y_t[i][j]);
        // find the likelihood of this particle
        w2[i][j] = sensor_measurement_model(z_t, x_ij_t);
      } // end loop j

      // cf http://www.cplusplus.com/reference/std/numeric/accumulate/
      Proba sum_w2_i = std::accumulate(w2[i], w2[i] + _N_y, 0);
      // for all persons (m), Y_t[i][j][m]   <-   Y_t[i][ĸ][m]
      // (just choose the best particle)
      for (int j = 0; j < _N_y; ++j) {
        unsigned int k =
            select_index_with_probas::select_given_sum
            (w2[i], w2[i] + _N_y, sum_w2_i);
        //ROS_WARN("i=pose particle #%i, j=person particle #%i, k:%i", i, j, k);
        for (int m = 0; m < _M; ++m) {
          _Y_t[i][j][m] = _Y_t[i][k][m];
        } // end loop m
      } // end loop j

      // store the weight (probability) of the whole particle set
      w1[i] = sum_w2_i;
    } // end loop i

    /*
     * build R_t, the new probable postions of the robot
     */
    for (int i = 0; i < _N_r; ++i) {
      unsigned int k =
          select_index_with_probas::select(w1, w1 + _N_r);
      _R_t[i] = _R_t[k];
    } // end loop i

    ROS_WARN_THROTTLE(1, "Time for algo(): %g ms.", _timer.time());
  } // end algo();

  //////////////////////////////////////////////////////////////////////////////

  //! the same as algo() except we use the stored values
  inline void algo_with_storage(const RobotCommandOrder & u_t,
                                const SensorMeasurement & z_t) {
    if (_R_t_minus1.size() == 0)
      init();
    algo(u_t, z_t);
    // store the new results
    _R_t_minus1 = _R_t;
    _Y_t_minus1 = _Y_t;
  } // end algo();

  //////////////////////////////////////////////////////////////////////////////

  virtual void init() = 0;

protected:
  Timer _timer;
  std::vector<RobotPose> _R_t_minus1, _R_t;
  ParticleVector3 _Y_t_minus1, _Y_t;

}; // end class ConditionalParticleFilter

} // end namespace  conditional_particle_filter_laser

#endif // CONDITIONALPARTICLEFILTER_H
