/*!
  \file        gtest_hist_tracking_skill.cpp
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2013/11/8

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
// Bring in gtest
#include <gtest/gtest.h>
#include "vision_utils/utils/timer.h"
#include "vision_utils/utils/matrix_testing.h"

#include <vision_utils/utils/rosmaster_alive.h>
#include <vision_utils/img_path.h>
#include "people_recognition_vision/hist_tracking_skill.h"

using namespace test_person_histogram_set_variables;
//#define DISPLAY

////////////////////////////////////////////////////////////////////////////////

static const int NO_USER = HistTrackingSkill::NO_USER_SELECTED;

inline int illus_cols(float nusers) { return nusers * PersonHistogramSet::CAPTION_WIDTH1; }
inline int illus_rows()             { return PersonHistogramSet::CAPTION_HEIGHT1 + image_utils::HEADER_SIZE; }
inline int illus_usercenter_x(int nuser) { return illus_cols(nuser+.5); }
inline int illus_usercenter_y()          { return illus_rows()/2; }

//! extend the visibilty of some parameters of HistTrackingSkill
class HistTrackingSkillExtended : public HistTrackingSkill {
public:
  inline void click_ref(int x, int y) {
    interface_mouse_cb(cv::EVENT_LBUTTONDOWN, x, y, 0, this);
  }
  inline void click_curr(int x, int y) {
    interface_mouse_cb(cv::EVENT_LBUTTONDOWN, x, illus_rows() + y, 0, this);
  }
  inline int get_curr_phs_prev_button() const { return curr_phs_prev_button; }
  inline int get_ref_phs_prev_button()  const { return ref_phs_prev_button; }
  void check_select_ref_curr(int exp_ref, int exp_curr) {
    ASSERT_TRUE(get_ref_phs_prev_button() == exp_ref)
        << "get_ref_phs_prev_button():" << get_ref_phs_prev_button() << ", exp_ref:" << exp_ref;
    ASSERT_TRUE(get_curr_phs_prev_button() == exp_curr)
        << "get_curr_phs_prev_button():" << get_curr_phs_prev_button() << ", exp_curr:" << exp_curr;
  }
  void check_images_sizes(int exp_ref, int exp_curr) {
    ASSERT_TRUE(matrix_testing::matrice_size_equal
                (_colormap_caption, 150, 300, 3, CV_8UC3));
    if (exp_ref> 0)
      ASSERT_TRUE(matrix_testing::matrice_size_equal
                  (_ref_phset_illus,
                   illus_cols(exp_ref+2), // include buttons
                   illus_rows(), 3, CV_8UC3));
    if (exp_curr > 0)
      ASSERT_TRUE(matrix_testing::matrice_size_equal
                  (_curr_phset_illus,
                   illus_cols(exp_curr),
                   illus_rows(), 3, CV_8UC3));
    if (exp_curr > 0 && exp_ref > 0)
      ASSERT_TRUE(matrix_testing::matrice_size_equal
                  (_interface,
                   illus_cols(std::max(exp_curr, exp_ref+2)), // include buttons
                   2*illus_rows(), 3, CV_8UC3));
  }
  void display() {
#ifdef DISPLAY
    illus();
    cv::waitKey(0);
#endif // DISPLAY
  }
};

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, empty_test) {
  if (!rosmaster_alive()) return;
  HistTrackingSkillExtended skill;
  ASSERT_NO_FATAL_FAILURE();
  ASSERT_TRUE(skill.nusers_ref() == 3);
  ASSERT_TRUE(skill.nhists_curr() == 0);
  skill.check_images_sizes(3, -1);
  skill.display();
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, assign_empty) {
  if (!rosmaster_alive()) return;
  HistTrackingSkillExtended skill;
  ASSERT_TRUE(skill.compare());
  PersonHistogramSet::PHMatch assign = skill.get_best_assign();
  skill.check_images_sizes(3, 0);
  unsigned int exp_assign_size = (0);
  ASSERT_TRUE(assign.size() == exp_assign_size)
      << "assign:" << assign.size() << ", exp_assign_size:" << exp_assign_size;
  for (unsigned int i = 0; i < exp_assign_size; ++i)
    ASSERT_TRUE(assign[i].second == assignment_utils::UNASSIGNED);
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, assign_n_users) {
  if (!rosmaster_alive()) return;
  // load some PersonHistogram
  HistTrackingSkillExtended skill;
  for (unsigned int nusers_curr = 0; nusers_curr < refset_hists_nb; ++nusers_curr) {
    // load curr_phset
    skill.get_curr_phset().clear();
    for (unsigned int i = 0; i < nusers_curr; ++i)
      ASSERT_TRUE(skill.get_curr_phset().push_back
                  (refset_filename_prefixes[i], refset_seeds[i], refset_kinect_serials[i], refset_labels()[i]));
    ASSERT_TRUE(skill.nhists_curr() == nusers_curr);
    // compare
    ASSERT_TRUE(skill.compare());
    skill.display();

    // checks
    skill.check_images_sizes(3, skill.nusers_curr());
    PersonHistogramSet::PHMatch assign = skill.get_best_assign();
    // check assign correct
    if (nusers_curr == 0) {
      ASSERT_TRUE(assign.empty());
      continue;
    }
    ASSERT_TRUE(assign.size() == refset_hists_nb)
        << "assign:" << assignment_utils::assignment_list_to_string(assign);
    assignment_utils::sort_assignment_list(assign);
    for (unsigned int i = 0; i < nusers_curr; ++i) {
      ASSERT_TRUE(assign[i].first == (int) i)
          << "assign:" << assignment_utils::assignment_list_to_string(assign);
      ASSERT_TRUE(assign[i].second == (int) i)
          << "assign:" << assignment_utils::assignment_list_to_string(assign);
    }
    // get label assign and check it
    PersonHistogramSet::LabelMatch label_assign = skill.get_best_label_assign();
    unsigned int exp_nlabels = skill.get_curr_phset().nlabels();
    ASSERT_TRUE(label_assign.size()  == exp_nlabels)
        << "label_assign:" << string_utils::map_to_string(label_assign);
    for (unsigned int label_idx = 1; label_idx <= exp_nlabels; ++label_idx)
      ASSERT_TRUE(label_assign[label_idx] == (int) label_idx)
          << "label_assign:" << string_utils::map_to_string(label_assign);
  } // end loop nusers_curr
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, gui_test) {
  if (!rosmaster_alive()) return;
  //HistTrackingSkill skill;
  HistTrackingSkillExtended skill;
  skill.check_select_ref_curr(NO_USER, NO_USER);
  // load some PersonHistogram
  skill.get_curr_phset().clear();
  unsigned int nhists_curr = 5;
  for (unsigned int i = 0; i < nhists_curr; ++i) {
    ASSERT_TRUE(skill.get_curr_phset().push_back
                (ainara_filename_prefixes[i], ainara_user_idx[i],
                 ainara_kinect_serials[i], ainara_labels[i]));
  } // end loop i
  ASSERT_TRUE(skill.nhists_curr() == nhists_curr);
  ASSERT_TRUE(skill.compare());
  skill.check_images_sizes(3, 1);

  printf("\n\n* clicks around ref*\n");
  // now simulate a click on ref user 0 -> select it
  skill.click_ref(illus_usercenter_x(0), illus_usercenter_y());
  skill.check_select_ref_curr(0, NO_USER);
  // click again on ref user 0 -> unselect it
  skill.click_ref(illus_usercenter_x(0), illus_usercenter_y());
  skill.check_select_ref_curr(NO_USER, NO_USER);
  // click on 0, then 1: should select 1
  skill.click_ref(illus_usercenter_x(0), illus_usercenter_y());
  skill.check_select_ref_curr(0, NO_USER);
  skill.click_ref(illus_usercenter_x(1), illus_usercenter_y());
  skill.check_select_ref_curr(1, NO_USER);
  skill.click_ref(illus_usercenter_x(1), illus_usercenter_y());
  skill.check_select_ref_curr(NO_USER, NO_USER);

  printf("\n\n* clicks around curr*\n");
  // now simulate a click on curr user 0 -> select it
  skill.click_curr(illus_usercenter_x(0), illus_usercenter_y());
  skill.check_select_ref_curr(NO_USER, 0);
  // click again on curr user 0 -> unselect it
  skill.click_curr(illus_usercenter_x(0), illus_usercenter_y());
  skill.check_select_ref_curr(NO_USER, NO_USER);
  // click on 0, then 1: should select 1
  skill.click_curr(illus_usercenter_x(0), illus_usercenter_y());
  skill.check_select_ref_curr(NO_USER, 0);
  skill.click_curr(illus_usercenter_x(1), illus_usercenter_y());
  skill.check_select_ref_curr(NO_USER, NO_USER); // no such user


  printf("\n\n* add a new PersonHistogram for an existing label*\n");
  // simulate a click on curr user 0 -> select it
  skill.click_curr(illus_usercenter_x(0), illus_usercenter_y());
  skill.check_select_ref_curr(NO_USER, 0);
  // click again on ref user 0 -> merge them and unselect everything
  unsigned int prev_nusers_ref = skill.nusers_ref(),
      prev_nhists_ref = skill.nhists_ref();
  skill.click_ref(illus_usercenter_x(1), illus_usercenter_y());
  skill.check_select_ref_curr(NO_USER, NO_USER);
  ASSERT_TRUE(skill.nusers_ref() == prev_nusers_ref)
      << "prev_nusers_ref:" << prev_nusers_ref << ", skill.nusers_ref():" << skill.nusers_ref();
  ASSERT_TRUE(skill.nhists_ref() == prev_nhists_ref+ 1)
      << "prev_nhists_ref:" << prev_nhists_ref << ", skill.nhists_ref():" << skill.nhists_ref();
  skill.display();
  skill.check_images_sizes(3, 1);
  // now in the other order: click on ref user, then curr user

  printf("\n\n* delete a ref user*\n");
  unsigned int delete_button_idx = skill.nusers_ref();
  prev_nusers_ref = skill.nusers_ref();
  // click on delete (should do nothing)
  skill.click_ref(illus_usercenter_x(delete_button_idx), illus_usercenter_y());
  skill.check_select_ref_curr(NO_USER, NO_USER);
  ASSERT_TRUE(skill.nusers_ref() == prev_nusers_ref)
      << "prev_nusers_ref:" << prev_nusers_ref << ", skill.nusers_ref():" << skill.nusers_ref();
  // select curr user 0 and click on delete (should do nothing)
  skill.click_curr(illus_usercenter_x(0), illus_usercenter_y());
  skill.check_select_ref_curr(NO_USER, 0);
  skill.click_ref(illus_usercenter_x(delete_button_idx), illus_usercenter_y());
  skill.check_select_ref_curr(NO_USER, NO_USER); // deselected
  ASSERT_TRUE(skill.nusers_ref() == prev_nusers_ref)
      << "prev_nusers_ref:" << prev_nusers_ref << ", skill.nusers_ref():" << skill.nusers_ref();
  // select ref user 0
  skill.click_ref(illus_usercenter_x(0), illus_usercenter_y());
  skill.check_select_ref_curr(0, NO_USER);
  // click on delete
  skill.click_ref(illus_usercenter_x(delete_button_idx), illus_usercenter_y());
  skill.check_select_ref_curr(NO_USER, NO_USER);
  // check nusers decreased
  ASSERT_TRUE(skill.nusers_ref() == prev_nusers_ref - 1)
      << "prev_nusers_ref:" << prev_nusers_ref << ", skill.nusers_ref():" << skill.nusers_ref();
  skill.display();
  skill.check_images_sizes(2, 1);

  printf("\n\n*add a ref user with a new label*\n");
  prev_nusers_ref = skill.nusers_ref();
  prev_nhists_ref = skill.nhists_ref();
  int create_button_idx=skill.nusers_ref()+ 1;
  // click on create (should do nothing)
  skill.click_ref(illus_usercenter_x(create_button_idx), illus_usercenter_y());
  skill.check_select_ref_curr(NO_USER, NO_USER);
  ASSERT_TRUE(skill.nusers_ref() == prev_nusers_ref)
      << "prev_nusers_ref:" << prev_nusers_ref << ", skill.nusers_ref():" << skill.nusers_ref();
  // select ref user 0 and click on create (should do nothing)
  skill.click_ref(illus_usercenter_x(1), illus_usercenter_y());
  skill.check_select_ref_curr(1, NO_USER);
  skill.click_ref(illus_usercenter_x(create_button_idx), illus_usercenter_y());
  skill.check_select_ref_curr(NO_USER, NO_USER); // deselected
  ASSERT_TRUE(skill.nusers_ref() == prev_nusers_ref)
      << "prev_nusers_ref:" << prev_nusers_ref << ", skill.nusers_ref():" << skill.nusers_ref();
  // select curr user 0
  int curr_selected_idx = 0;
  skill.click_curr(illus_usercenter_x(curr_selected_idx), illus_usercenter_y());
  skill.check_select_ref_curr(NO_USER, curr_selected_idx);
  // click on create
  skill.click_ref(illus_usercenter_x(create_button_idx), illus_usercenter_y());
  skill.check_select_ref_curr(NO_USER, NO_USER);
  // check nusers increased
  ASSERT_TRUE(skill.nhists_ref() == prev_nhists_ref + 1)
      << "prev_nhists_ref:" << prev_nhists_ref << ", skill.nhists_ref():" << skill.nhists_ref();
  ASSERT_TRUE(skill.nusers_ref() == prev_nusers_ref + 1)
      << "prev_nusers_ref:" << prev_nusers_ref << ", skill.nusers_ref():" << skill.nusers_ref();
  skill.display();
  skill.check_images_sizes(3, 1);
  // check the PersonHistogram are equal
  const PersonHistogram
      *selected_ph = &(skill.get_curr_phset().at(skill.nhists_curr()-1)),
      *created_ph = &(skill.get_ref_phset().at(skill.nhists_ref()-1));
  ASSERT_TRUE(created_ph->get_input_images_nb() == 1);
  ASSERT_NEAR(selected_ph->compare_to(*created_ph), 0, 1E-2);
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv){
  ros::init(argc, argv, "gtest_hist_tracking_skill");
  // Run all the tests that were declared with TEST()
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
