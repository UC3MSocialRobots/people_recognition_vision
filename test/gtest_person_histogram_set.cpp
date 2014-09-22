/*!
  \file        gtest_person_histogram_set.cpp
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2013/11/5

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
 */
// Bring in gtest
#include <gtest/gtest.h>
// people_msgs
#include "src/person_histogram_set/person_histogram_set.h"
#include <vision_utils/databases_io/test_person_histogram_set_variables.h>
#include "src/time/timer.h"
#include "src/test/matrix_testing.h"

using namespace test_person_histogram_set_variables;
//#define DISPLAY

TEST(TestSuite, to_mat) {
  PersonHistogramSet phset;
  ASSERT_TRUE(phset.nhists() == 0);
  ASSERT_TRUE(phset.push_back_vec(refset_filename_prefixes, refset_seeds, refset_kinect_serials, refset_labels()));
  unsigned int nhists = refset_filename_prefixes.size();
  ASSERT_TRUE(phset.nhists() == nhists);

  cv::Mat ph2mat;
  ASSERT_TRUE(phset.to_mat(ph2mat));
  ASSERT_TRUE(ph2mat.cols == PersonHistogramSet::MAT_COLS);
#ifndef SVM_USE_STD_DEV_HIST
  for (unsigned int ph_idx = 0; ph_idx < nhists; ++ph_idx) {
    ASSERT_TRUE(matrix_testing::matrices_equal
                (phset.at(ph_idx).get_hist_vector().at(1).t(),
                 ph2mat.row(ph_idx)));
  } // end loop ph_idx
#endif
  ASSERT_TRUE(ph2mat.rows == (int) nhists);
}

////////////////////////////////////////////////////////////////////////////////

template<class T1, class T2, class T3>
inline void test_io(const std::vector<T1> & arg1,
                    const std::vector<T2> & arg2,
                    const std::vector<T3> & arg3,
                    const std::vector<PersonHistogramSet::PersonLabel> & labels) {
  PersonHistogramSet phset;
  ASSERT_TRUE(phset.push_back_vec(arg1, arg2, arg3, labels));
  ASSERT_TRUE(phset.train_svm());
  ASSERT_TRUE(phset.to_yaml("/tmp/FooPersonHistogramSet.yaml"));

  PersonHistogramSet phset2;
  ASSERT_TRUE(phset2.from_yaml("/tmp/FooPersonHistogramSet.yaml"));

  // simple tests
  ASSERT_TRUE(phset2.nhists() == phset.nhists());
  ASSERT_TRUE(phset2.nlabels() == phset.nlabels());
  for (unsigned int i = 0; i < phset2.nhists(); ++i)
    ASSERT_TRUE(phset.label(i) == phset2.label(i));

  // assert images not empty
  for (unsigned int i = 0; i < phset2.nhists(); ++i) {
    ASSERT_TRUE(matrix_testing::matrices_equal(phset.at(i).get_illus_color_img(),
                                               phset2.at(i).get_illus_color_img()));
#ifdef DISPLAY
    cv::imshow("illus1", phset.at(i).get_illus_color_mask());
    cv::imshow("illus2", phset2.at(i).get_illus_color_mask());
    cv::waitKey(0);
#endif // DISPLAY
    cv::Mat1b frame_diff;
    ASSERT_NEAR(matrix_testing::rate_of_changes_between_two_images
                (phset.at(i).get_illus_color_mask(), phset2.at(i).get_illus_color_mask(),
                 frame_diff, 1), 0, 1E-2);
  }

  // hists
  assignment_utils::MatchList assign;
  ASSERT_TRUE(phset.compare_to(phset2, assign, true, true, CV_COMP_BHATTACHARYYA, false));
  ASSERT_TRUE(assign.size() == phset.nhists());
  for (unsigned int i = 0; i < phset2.nhists(); ++i) {
    ASSERT_TRUE(assign[i].first == assign[i].second);
    ASSERT_NEAR(assign[i].cost, 0, 1E-2)
        << "assign:" << assignment_utils::assignment_list_to_string(assign);
  } // end loop i

  // SVM test
  if (phset.nlabels() >= 2) {
    for (unsigned int i = 0; i < phset2.nhists(); ++i) {
      int out_label, exp_label = phset.label(i);
      ASSERT_TRUE(phset2.compare_svm(phset.at(i), out_label));
      ASSERT_TRUE(out_label == exp_label)
          << "out_label:" << out_label << ", expected_label:" << exp_label;
    } // end loop i
  } // end if (phset.nlabels() >= 2)

#ifdef DISPLAY
  cv::imshow("dist_matrix_illus", phset.get_dist_matrix_illus());
  cv::imshow("dist_matrix_colormap_caption", phset.get_dist_matrix_colormap_caption());
  cv::imshow("dist_matrix_illus_caption1", phset.get_dist_matrix_illus_caption1());
  cv::imshow("dist_matrix_illus_caption2", phset.get_dist_matrix_illus_caption2());
  cv::waitKey(0);
#endif // DISPLAY
}

TEST(TestSuite, to_yaml_juggling) { test_io(juggling_filename_prefixes, juggling_seeds, juggling_kinect_serials, juggling_labels); }
TEST(TestSuite, to_yaml_all)      { test_io(refset_filename_prefixes, refset_seeds, refset_kinect_serials, refset_labels()); }
TEST(TestSuite, to_yaml_david)    { test_io(david_filename_prefixes, david_user_idx, david_kinect_serials, david_labels); }

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, generate_headers_empty) {
  PersonHistogramSet phset;
  std::vector<std::string> labels, exp_headers;
  phset.generate_headers(titlemaps::int_to_uppercase_letter, labels);
  ASSERT_TRUE(labels == exp_headers);
}

template<class T1, class T2, class T3>
inline void test_generate_headers(const std::vector<T1> & arg1,
                                  const std::vector<T2> & arg2,
                                  const std::vector<T3> & arg3,
                                  const std::vector<PersonHistogramSet::PersonLabel> & labels,
                                  const std::vector<std::string> & exp_headers) {
  PersonHistogramSet phset;
  ASSERT_TRUE(phset.push_back_vec(arg1, arg2, arg3, labels));
  std::vector<std::string> headers;
  phset.generate_headers(titlemaps::int_to_uppercase_letter, headers);
  ASSERT_TRUE(headers == exp_headers)
      << "exp_headers:" << StringUtils::iterable_to_string(exp_headers)
      << ", labels:" << StringUtils::iterable_to_string(headers);
}

TEST(TestSuite, generate_headers_david) {
  std::vector<std::string> exp_headers;
  for (unsigned int i = 0; i < david_hists_nb; ++i)
    exp_headers.push_back("A" + StringUtils::cast_to_string(i+1));
  test_generate_headers(david_filename_prefixes, david_user_idx,
                        david_kinect_serials, david_labels, exp_headers);
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, generate_headers_juggling) {
  std::vector<std::string> exp_headers;
  for (unsigned int i = 0; i < juggling_hists_nb; ++i)
    exp_headers.push_back("A" + StringUtils::cast_to_string(i+1));
  test_generate_headers(juggling_filename_prefixes, juggling_seeds,
                        juggling_kinect_serials, juggling_labels, exp_headers);
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, generate_headers_all) {
  std::vector<std::string> exp_headers;
  for (unsigned int i = 0; i < juggling_hists_nb; ++i)
    exp_headers.push_back("A" + StringUtils::cast_to_string(i+1));
  for (unsigned int i = 0; i < alberto_hists_nb; ++i)
    exp_headers.push_back("B" + StringUtils::cast_to_string(i+1));
  for (unsigned int i = 0; i < alvaro_hists_nb; ++i)
    exp_headers.push_back("C" + StringUtils::cast_to_string(i+1));
  test_generate_headers(refset_filename_prefixes, refset_seeds,
                        refset_kinect_serials, refset_labels(), exp_headers);
}

////////////////////////////////////////////////////////////////////////////////

void test_train_svm(bool add_ainara = false,
                    bool add_david = false,
                    bool add_arnaud = false) {
  // load a fewhistogram
  PersonHistogramSet phset;
  ASSERT_TRUE(phset.nhists() == 0);

  ASSERT_TRUE(phset.push_back_vec(refset_filename_prefixes, refset_seeds, refset_kinect_serials, refset_labels()));
  ASSERT_TRUE(phset.nhists() == refset_hists_nb);

  if (add_ainara) { // adding all ainara histograms
    ASSERT_TRUE(phset.push_back_vec(ainara_filename_prefixes, ainara_user_idx,
                                    ainara_kinect_serials, ainara_labels));
  } // end add_ainara

  if (add_david) { // adding all david histograms
    ASSERT_TRUE(phset.push_back_vec(david_filename_prefixes, david_user_idx,
                                    david_kinect_serials, david_labels));
  } // end add_david

  if (add_arnaud) { // adding all arnaud histograms
    ASSERT_TRUE(phset.push_back_vec(arnaud_filename_prefixes, arnaud_user_idx,
                                    arnaud_kinect_serials, arnaud_labels));
  } // end add_arnaud

  // train
  Timer timer;
  ASSERT_TRUE(phset.train_svm());
  timer.printTime("train_svm()");
#ifdef DISPLAY
  phset.svm2img();
#endif // DISPLAY

  // test
  unsigned int nsamples = phset.nhists();
  timer.reset();
  for (unsigned int i = 0; i < nsamples; ++i) {
    int out_label, exp_label = phset.label(i);
    ASSERT_TRUE(phset.compare_svm(phset.at(i), out_label));
    //  std::cout << "out_label:" << out_label
    //            << ", expected_label:" << exp_label << std::endl;
    ASSERT_TRUE(out_label == exp_label)
        << "out_label:" << out_label << ", expected_label:" << exp_label;
  } // end loop i
  timer.printTime_factor("compare_svm", nsamples);
} // end test_train_svm();

TEST(TestSuite, train_svm_3users) { test_train_svm(); }
TEST(TestSuite, train_svm_4users) { test_train_svm(true); }
TEST(TestSuite, train_svm_5users) { test_train_svm(true, true); }
TEST(TestSuite, train_svm_6users) { test_train_svm(true, true, true); }

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv){
  // Run all the tests that were declared with TEST()
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
