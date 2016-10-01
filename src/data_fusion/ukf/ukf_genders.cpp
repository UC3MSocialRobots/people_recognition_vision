/*!
  \file        ukf_genders.cpp
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2013/1

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

\section Parameters
  - \b "~ppl_input_topic"
        [string] (default: "ppl")
        Where the face recognition results will be obtained.

\section Subscriptions
  - \b ${ppl_input_topic}
        [people_msgs_rl::PeoplePoseList]
        The found faces ROIs and the name of the persons recognized

 */


#include <std_msgs/Int16.h>
#include <ros/ros.h>
// AD
#include "vision_utils/utils/combinatorics_utils.h"
#include <easykf-2.03/src/ukf.h>
#include <gnuplot-cpp/gnuplot_i.hpp>
#include "vision_utils/utils/system_utils.h"
using namespace ukf::state;
// people_msgs_rl
#include "people_msgs_rl/PeoplePoseList.h"


class UkfGenders {
public:
  UkfGenders() {
    /*
     * UKF initialization
     */
    // The parameters for the evolution equation
    s.params = gsl_vector_alloc(1);

    // Initialization of the parameters
    p.n = 2; // Size of the state vector
    p.no = 2; // Dimension of the output : the measurements
    p.kpa = 0.0; // kappa >= 0, kappa = 0 is a good choice. According to van der Merwe, its value is not critical
    p.alpha = 0.9; // alpha <= 1, "Size" of sigma-point distribution. Should be small if the function is strongly non-linear
    p.beta = 2.0; // Non negative weights used to introduce knowledge about the higher order moments of the distribution. For gaussian distributions, beta = 2 is a good choice

    //EvolutionNoise * evolution_noise = new EvolutionAnneal(1e-1, 1.0, 1e-2);
    //EvolutionNoise * evolution_noise = new EvolutionRLS(1e-5, 0.9995);
    EvolutionNoise * evolution_noise = new EvolutionRobbinsMonro(1e-4, 1e-3);
    p.evolution_noise = evolution_noise;

    p.measurement_noise = 1.0; // Covariance of the observation noise
    p.prior_x= 1.0; // Prior estimate of the covariance matrix of the state

    // Initialization of the state and parameters
    ukf_init(p,s);

    // Allocate the input/output vectors
    yi = gsl_vector_alloc(p.no);
    gsl_vector_set_zero(yi);

    // set initial value for the state, in s.xi
    gsl_vector_set(s.xi, 0, 0);

    // ROS init
    ros::NodeHandle nh_public, nh_private("~");
    std::string ppl_input_topic = "ppl";
    nh_private.param("ppl_input_topic",
                     ppl_input_topic,
                     ppl_input_topic);
    _face_recognition_results_sub = nh_public.subscribe
        (ppl_input_topic, 1,
         &UkfGenders::face_reco_result_cb, this);
    _men_pub = nh_public.advertise<std_msgs::Int16>("GENDERS_MEN_DETECTED", 1);
    _women_pub = nh_public.advertise<std_msgs::Int16>("GENDERS_WOMEN_DETECTED", 1);
  }

  //////////////////////////////////////////////////////////////////////////////

  void measurement_update(const unsigned int nb_men, const unsigned int nb_women) {
    // update yi
    gsl_vector_set(yi, 0, nb_men);
    gsl_vector_set(yi, 1, nb_women);

    // if we need some params in the evo function, store them in s.params->data
    // ...

    // Provide the observation and iterate
    // ukf_param &p, ukf_state &s, FunctProcess f, FunctObservation h, gsl_vector* yi
    ukf_iterate(p, s, evo_function, obs_function, yi);

    // the new estimated state is in s.xi
    // ...
  }

  //////////////////////////////////////////////////////////////////////////////

  //! Evolution function - xk_1 = previous state, xk = new one (to fill)
  static void evo_function(gsl_vector * params, gsl_vector * xk_1, gsl_vector * xk)
  {
    // copy xk_1 -> xk (no evolution)
    for(unsigned int i = 0 ; i < xk->size ; ++i)
      gsl_vector_set(xk, i, gsl_vector_get(xk_1,i));
  }

  //////////////////////////////////////////////////////////////////////////////

  //! Observation function: we suppose a state xk and need to fill yk
  static void obs_function(gsl_vector * xk , gsl_vector * yk)
  {
    // copy xk -> yk (observations are the sames as state)
    for(unsigned int i = 0 ; i < yk->size ; ++i)
      gsl_vector_set(yk, i, gsl_vector_get(xk,i));
  }

  //////////////////////////////////////////////////////////////////////////////

  inline unsigned int get_estimated_nb_men() const {
    return round(gsl_vector_get(s.xi, 0));
  }
  inline unsigned int get_estimated_nb_women() const {
    return round(gsl_vector_get(s.xi, 1));
  }

  //////////////////////////////////////////////////////////////////////////////

  void face_reco_result_cb
  (const people_msgs_rl::PeoplePoseListConstPtr & msg) {
    // count nb of men and women
    int nb_men = 0, nb_women = 0;
    std::ostringstream all_names;
    for (unsigned int reco_idx = 0; reco_idx < msg->poses.size(); ++reco_idx) {
      std::string current_name = msg->poses[reco_idx].person_name;
      all_names << current_name << "; ";
      if (current_name == "man" || current_name == "boy")
        ++nb_men;
      else if (current_name == "woman" || current_name == "girl")
        ++nb_women;
    } // end loop reco_idx
    if  (nb_men == 0 && nb_women == 0) {

      ROS_WARN_THROTTLE(1, "The reco message with names '%s' does not content gender info.",
                        all_names.str().c_str());
      return;
    }
    // now update model
    measurement_update(nb_men, nb_women);
    // now publish estimated value
    std_msgs::Int16 msg_out;
    msg_out.data = get_estimated_nb_men();
    _men_pub.publish(msg_out);
    msg_out.data = get_estimated_nb_women();
    _women_pub.publish(msg_out);
  } // end face_reco_result_cb();


  ros::NodeHandle nh;
  ukf_param p; //!< param vector
  ukf_state s; //!< estimated state
  gsl_vector * yi; //!< observation vector

  //! face reco sub
  ros::Subscriber _face_recognition_results_sub;
  ros::Publisher _men_pub, _women_pub;
};

////////////////////////////////////////////////////////////////////////////////

void ukf_test() {
  srand(time(NULL));
  UkfGenders ukf_genders;
  std::vector<int> men_ground_truth, men_noisy_measurement, men_gnu,
      women_ground_truth, women_noisy_measurement, women_gnu;

  for (int iter = 0; iter < 30; ++iter) {
    int this_men_ground_truth = (iter < 12 ? 1 : 2);
    int this_men_noisy_measurement = std::max
        (0, this_men_ground_truth +
         (int) combinatorics_utils::rand_gaussian());
    int this_women_ground_truth = (iter < 10 ? 2 : 1);
    int this_women_noisy_measurement = std::max
        (0, this_women_ground_truth +
         (int) combinatorics_utils::rand_gaussian());
    ukf_genders.measurement_update(this_men_noisy_measurement, this_women_noisy_measurement);

    men_ground_truth.push_back(this_men_ground_truth);
    women_ground_truth.push_back(this_women_ground_truth);
    men_noisy_measurement.push_back(this_men_noisy_measurement);
    women_noisy_measurement.push_back(this_women_noisy_measurement);
    men_gnu.push_back(ukf_genders.get_estimated_nb_men());
    women_gnu.push_back(ukf_genders.get_estimated_nb_women());
  } // end loop iter

  Gnuplot plotter;
  plotter.set_style("lines lw 2").plot_x(men_ground_truth, "men (ground truth)");
  plotter.set_style("lines lw 1").plot_x(men_noisy_measurement, "men (noisy measurement)");
  plotter.set_style("lines lw 1").plot_x(men_gnu, "men (UKF result)");

  Gnuplot plotter2;
  plotter2.set_style("lines lw 2").plot_x(women_ground_truth, "women (ground truth)");
  plotter2.set_style("lines lw 1").plot_x(women_noisy_measurement, "women (noisy measurewoment)");
  plotter2.set_style("lines lw 1").plot_x(women_gnu, "women (UKF result)");
  system_utils::wait_for_key();
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  ros::init(argc, argv, "ukf_genders");
  ukf_test();
  //  UkfGenders ukf_genders;
  //  ros::spin();
  return 0;
}
