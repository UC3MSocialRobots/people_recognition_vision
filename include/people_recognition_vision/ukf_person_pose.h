/*!
  \file        ukf_person_pose.h
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2012/10/14

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

A class to estimate the motion of the users
thanks to Unscented Kalman Filters.
*/
#ifndef UKF_PERSON_POSE_H
#define UKF_PERSON_POSE_H

#include <easykf-2.03/src/ukf.h>
#include <vision_utils/rand_gaussian.h>

using namespace ukf::state;

class UkfPersonPose {
public:
  template<class Pt3>
  UkfPersonPose(const Pt3 & initial_pos, const double & orien, const double & speed) {
    // The parameters for the evolution equation
    s.params = gsl_vector_alloc(1);

    // Initialization of the parameters
    p.n = 5; // Size of the state vector
    p.no = 3; // Dimension of the output : the measurements
    p.kpa = 0.0; // kappa >= 0, kappa = 0 is a good choice. According to van der Merwe, its value is not critical
    p.alpha = 0.9; // alpha <= 1, "Size" of sigma-point distribution. Should be small if the function is strongly non-linear
    p.beta = 2.0; // Non negative weights used to introduce knowledge about the higher order moments of the distribution. For gaussian distributions, beta = 2 is a good choice

    EvolutionNoise * evolution_noise = new EvolutionAnneal(1e-1, 1.0, 1e-2);
    //EvolutionNoise * evolution_noise = new EvolutionRLS(1e-5, 0.9995);
    //EvolutionNoise * evolution_noise = new EvolutionRobbinsMonro(1e-4, 1e-3);
    p.evolution_noise = evolution_noise;

    // Covariance of the observation noise
    p.measurement_noise = .5;
    p.prior_x= 1.0;

    // Initialization of the state and parameters
    ukf_init(p,s);

    // Allocate the input/output vectors
    yi = gsl_vector_alloc(p.no);
    gsl_vector_set_zero(yi);
    set_position(initial_pos, orien, speed);
  } // end ctor

  //////////////////////////////////////////////////////////////////////////////

  template<class Pt3>
  void measurement_update(const Pt3 & face_detec,
                          const double & delta_t_sec) {
    s.params->data[0] = delta_t_sec;

    gsl_vector_set(yi, 0, face_detec.x);
    gsl_vector_set(yi, 1, face_detec.y);
    gsl_vector_set(yi, 2, face_detec.z);

    // Provide the observation and iterate
    // ukf_param &p, ukf_state &s, FunctProcess f, FunctObservation h, gsl_vector* yi
    ukf_iterate(p, s, evo_function, obs_function, yi);
  }

  //////////////////////////////////////////////////////////////////////////////

  template<class Pt3>
  inline void set_position(const Pt3 & face_detec,
                           const double & orien, const double & speed) {
    gsl_vector_set(s.xi, 0, face_detec.x);
    gsl_vector_set(s.xi, 1, face_detec.y);
    gsl_vector_set(s.xi, 2, face_detec.z);
  }

  //////////////////////////////////////////////////////////////////////////////

  template<class Pt3>
  inline void get_position(Pt3 & face_detec) const {
    face_detec.x = gsl_vector_get(s.xi, 0);
    face_detec.y = gsl_vector_get(s.xi, 1);
    face_detec.z = gsl_vector_get(s.xi, 2);
  }

  //////////////////////////////////////////////////////////////////////////////

  template<class Pt3>
  inline void get_state(Pt3 & face_detec, double & orien, double & speed) const {
    get_position(face_detec);
    orien = gsl_vector_get(s.xi, 3);
    speed = gsl_vector_get(s.xi, 4);
  }

  //////////////////////////////////////////////////////////////////////////////

  //! get the standard deviation (meters)
  inline double get_std_dev() const {
    //    for (int i = 0; i < p.n; ++i) {
    //      for (int j = 0; j < p.n; ++j)
    //        std::cout << gsl_matrix_get(s.Pxxi, i, j) << ",\t ";
    //      std::cout << std::endl;
    //    } // end loop i
    // gsl_matrix_fprintf(stdout,s.Pxxi,"%f");

    // get the average of the 3 first diagonal values of the state covariance matrix
    // why 3? (x, y, z), no interest in orientation and speed
    double average = 0;
    for (int i = 0; i < 3; ++i)
      average += fabs(gsl_matrix_get(s.Pxxi, i, i));
    average = average / 3;
    return average;
  }

  //////////////////////////////////////////////////////////////////////////////

private:
  static const double brownian_speed_amp = .1;

  // Evolution function
  static void evo_function(gsl_vector * params, gsl_vector * xk_1, gsl_vector * xk)
  {
    double x = gsl_vector_get(xk_1,0);
    double y = gsl_vector_get(xk_1,1);
    double z = gsl_vector_get(xk_1,2);
    double orien = gsl_vector_get(xk_1,3);
    double speed = gsl_vector_get(xk_1,4);
    double dt = gsl_vector_get(params, 0);

    // evolution law
    gsl_vector_set(xk, 0, x + dt * speed * cos(orien));
    gsl_vector_set(xk, 1, y + dt * speed * sin(orien));
    gsl_vector_set(xk, 2, z);
    // set new brownian direction
    gsl_vector_set(xk, 3, 2 * M_PI * drand48());
    // gsl_vector_set(xk, 3, 2 * M_PI * vision_utils::rand_gaussian());
    // and speed in [0 .. brownian_speed_amp]
    gsl_vector_set(xk, 4, brownian_speed_amp * vision_utils::rand_gaussian());
  }

  // Observation function
  static void obs_function(gsl_vector * xk , gsl_vector * yk)
  {
    for(unsigned int i = 0 ; i < yk->size ; ++i)
      gsl_vector_set(yk, i, gsl_vector_get(xk,i));
  }

  ukf_param p;
  ukf_state s;
  gsl_vector * yi;
}; // end class UkfPersonPose

#endif // UKF_PERSON_POSE_H
