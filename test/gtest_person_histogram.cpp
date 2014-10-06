/*!
  \file        gtest_person_histogram.cpp
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2013/11/6

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
#include "person_histogram_set/person_histogram.h"
#include <databases_io/test_person_histogram_set_variables.h>
#include "time/timer.h"
#include "test/matrix_testing.h"

using namespace test_person_histogram_set_variables;
//#define DISPLAY

void ASSERT_SOUND_PH(const PersonHistogram & ph,
                     int exp_input_images_nb = -1) {
  // check sizes of hist vector
  if (exp_input_images_nb >= 0)
    ASSERT_TRUE(ph.get_input_images_nb() == exp_input_images_nb);
  ASSERT_TRUE(ph.get_hist_vector().size() == PersonHistogram::BODY_PARTS);
  for (unsigned int hidx = 0; hidx < PersonHistogram::BODY_PARTS; ++hidx) {
    ASSERT_TRUE(ph.get_hist_vector()[hidx].rows == PersonHistogram::HIST_NBINS);
    ASSERT_TRUE(cv::sum(ph.get_hist_vector()[hidx])[0] > 0); // non empty hist
  }

  // check multimask values = 1, 2, 3
  std::vector<uchar> values;
  image_utils::get_all_different_values(ph.get_multimask(), values, true);
  ASSERT_TRUE(values.size() == PersonHistogram::BODY_PARTS);
  for (unsigned int i = 1; i <= PersonHistogram::BODY_PARTS; ++i)
    ASSERT_TRUE(std::find(values.begin(), values.end(), (uchar) i)
                != values.end()) << "values:" << StringUtils::iterable_to_int_string(values);

  // check illus images
  int cols = ph.get_illus_color_img().cols, rows = ph.get_illus_color_img().rows;
  ASSERT_TRUE(cols > 0 && rows > 0) << "cols:" << cols << ", rows:" << rows;
  ASSERT_TRUE(matrix_testing::matrice_size_equal
              (ph.get_illus_color_img(), cols, rows, 3, CV_8UC3));
  ASSERT_TRUE(matrix_testing::matrice_size_equal
              (ph.get_illus_color_mask(), cols, rows, 1, CV_8U));
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, test_copy) {
  PersonHistogram ph, ph_copy;
  ASSERT_TRUE(ph.create(juggling1_file, juggling1_seed, KINECT_SERIAL_LAB()));
  ASSERT_SOUND_PH(ph, 1);
  ph_copy = ph;
  ASSERT_SOUND_PH(ph_copy, 1);
}

////////////////////////////////////////////////////////////////////////////////

template<class T2, class T3>
void test_io_PersonHistogram(const std::string & filename_prefix,
                             const T2 & arg2,
                             const T3 & arg3) {
  // compute all histogams
  Timer timer;
  PersonHistogram ph(filename_prefix, arg2, arg3), ph2;
  timer.printTime("loading data and computing histograms");

  timer.reset();
  image_utils::to_yaml(ph, "/tmp/foo", "PersonHistogram");
  timer.printTime("to yaml");
  image_utils::from_yaml(ph2, "/tmp/foo", "PersonHistogram");
  timer.printTime("to yaml -> from yaml");

  // evaluate dist between hists
  double dist = ph.compare_to(ph2);
  ASSERT_NEAR(dist, 0, 1E-2) << "dist:" << dist;

#ifdef DISPLAY
  cv::imshow("ph_color", ph.get_illus_color_img());
  cv::imshow("ph2_color", ph2.get_illus_color_img());
  cv::imshow("ph_mask", ph.get_illus_color_mask());
  cv::imshow("ph2_mask", ph2.get_illus_color_mask());
  ph.show_illus_image(0);
#endif // DISPLAY

  // assert masks are equal
  cv::Mat1b frame_diff;
  ASSERT_NEAR(matrix_testing::rate_of_changes_between_two_images
              (ph.get_illus_color_mask(), ph2.get_illus_color_mask(),
               frame_diff, 1), 0, 1E-2);
}

TEST(TestSuite, test_io_PersonHistogram_all_files) {
  for (unsigned int i = 0; i < refset_hists_nb; ++i)
    test_io_PersonHistogram(refset_filename_prefixes[i], refset_seeds[i], refset_kinect_serials[i]);
  for (unsigned int i = 0; i < ainara_hists_nb; ++i)
    test_io_PersonHistogram(ainara_filename_prefixes[i], ainara_user_idx[i], ainara_kinect_serials[i]);
  for (unsigned int i = 0; i < david_hists_nb; ++i)
    test_io_PersonHistogram(david_filename_prefixes[i], david_user_idx[i], david_kinect_serials[i]);
  for (unsigned int i = 0; i < arnaud_hists_nb; ++i)
    test_io_PersonHistogram(arnaud_filename_prefixes[i], arnaud_user_idx[i], arnaud_kinect_serials[i]);
}

////////////////////////////////////////////////////////////////////////////////

template<class T2, class T3>
void test_read_from_file(const std::string & filename_prefix,
                         const T2 & arg2,
                         const T3 & arg3) {
  printf("test_read_from_file('%s')\n", filename_prefix.c_str());
  PersonHistogram ph;
  ASSERT_TRUE(ph.create(filename_prefix, arg2, arg3));
#ifdef DISPLAY
  cv::imshow("illus_color_img", ph.get_illus_color_img());
  cv::imshow("illus_color_mask", ph.get_illus_color_mask());
  ph.show_illus_image(0);
#endif // DISPLAY
  ASSERT_SOUND_PH(ph, 1);
}

TEST(TestSuite, test_read_from_file_all_files) {
  for (unsigned int i = 0; i < refset_hists_nb; ++i)
    test_read_from_file(refset_filename_prefixes[i], refset_seeds[i], refset_kinect_serials[i]);
  for (unsigned int i = 0; i < ainara_hists_nb; ++i)
    test_read_from_file(ainara_filename_prefixes[i], ainara_user_idx[i], ainara_kinect_serials[i]);
  for (unsigned int i = 0; i < david_hists_nb; ++i)
    test_read_from_file(david_filename_prefixes[i], david_user_idx[i], david_kinect_serials[i]);
  for (unsigned int i = 0; i < arnaud_hists_nb; ++i)
    test_read_from_file(arnaud_filename_prefixes[i], arnaud_user_idx[i], arnaud_kinect_serials[i]);
}

////////////////////////////////////////////////////////////////////////////////

void test_rgb_fill(const cv::Mat3b & new_rgb, std::string exp_color) {
  cv::Mat1b user_mask;
  cv::Mat rgb, depth;
  int image_idx = rand() % ainara_hists_nb;
  ASSERT_TRUE(image_utils::read_rgb_depth_user_image_from_image_file
              (ainara_filename_prefixes[image_idx], &rgb, &depth, &user_mask));
  user_mask = (user_mask == ainara_user_idx[image_idx]);
  PersonHistogram ph;
  cv::resize(new_rgb, rgb, rgb.size());
  ASSERT_TRUE(ph.create(rgb, user_mask, depth, ainara_kinect_serials[image_idx]));
#ifdef DISPLAY
  ph.show_illus_image(0);
  cv::imshow("rgb", rgb);
  cv::imshow("depth", image_utils::depth2viz(depth));
  cv::imshow("multimask", user_image_to_rgb(ph.get_multimask()));
  cv::imshow("user_mask", user_mask);
  ph.show_illus_image(0);
#endif // DISPLAY

  ASSERT_SOUND_PH(ph, 1);
  for (unsigned int hidx = 0; hidx < PersonHistogram::BODY_PARTS; ++hidx) {
    std::string hist_dominant_color = histogram_utils::hue_hist_dominant_color_to_string
                                      (ph.get_hist_vector()[hidx]);
    ASSERT_TRUE(hist_dominant_color == exp_color)
        << "hist_dominant_color:" << hist_dominant_color << ", exp_color:" << exp_color;
  }
}

void test_rgb_fill(cv::Scalar color, std::string exp_color) {
  test_rgb_fill(cv::Mat3b(480, 640, cv::Vec3b(color[0], color[1], color[2])), exp_color);
}

/* There are 15 bins, so the hue step from bin to bin is 180/15 = 12 = 24 degrees.
 * The resulting OpenCV hue in [0, 128] is int (hue380 / 24) * 12.
 * Take that into account for the color names!
 */
//TEST(TestSuite, test_rgb_fill_black)  { test_rgb_fill(CV_RGB(0, 0, 0), "black"); }
//TEST(TestSuite, test_rgb_fill_white)  { test_rgb_fill(CV_RGB(255, 255, 255), "white"); }
TEST(TestSuite, test_rgb_fill_red)    { test_rgb_fill(CV_RGB(255, 0, 0), "red"); }
TEST(TestSuite, test_rgb_fill_blue)   { test_rgb_fill(CV_RGB(0, 0, 255), "blue"); }
TEST(TestSuite, test_rgb_fill_orange) { test_rgb_fill(CV_RGB(255, 128, 0), "orange"); }
//TEST(TestSuite, test_rgb_fill_yellow) { test_rgb_fill(CV_RGB(255, 255, 0), "yellow"); }
TEST(TestSuite, test_rgb_fill_yellow) { test_rgb_fill(CV_RGB(255, 255, 0), "orange"); }
TEST(TestSuite, test_rgb_fill_lemon)  { test_rgb_fill(CV_RGB(128, 255, 0), "lemon"); }
TEST(TestSuite, test_rgb_fill_green)  { test_rgb_fill(CV_RGB(0, 255, 0), "green"); }
TEST(TestSuite, test_rgb_fill_purple) { test_rgb_fill(CV_RGB(255, 0, 255), "purple"); }

TEST(TestSuite, test_rgb_multi) {
  int rows = 480, cols = 640;
  cv::Mat3b rgb(rows, cols);
  for (int row = 0; row < rows; ++row) {
    cv::Vec3b* data = rgb.ptr<cv::Vec3b>(row);
    for (int col = 0; col < cols; ++col)
      data[col] = color_utils::hue2rgb<cv::Vec3b>(rand() % 180);
  } // end loop row
  test_rgb_fill(rgb, "multicolor");
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, to_mat_empty) {
  PersonHistogram ph;
  cv::Mat ph2mat;
  // PersonHistogram not allocated: should fail
  ASSERT_FALSE(ph.to_mat(ph2mat));
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, to_mat) {
  PersonHistogram ph(juggling1_file, juggling1_seed, KINECT_SERIAL_LAB());
  cv::Mat ph2mat;
  //matrix not allocated: should succeed
  ASSERT_TRUE(ph.to_mat(ph2mat));
  // bad type
  ph2mat.create(1, PersonHistogram::MAT_COLS, CV_8U);
  ASSERT_FALSE(ph.to_mat(ph2mat));
  // now correct call
  ph2mat.release();
  ASSERT_TRUE(ph.to_mat(ph2mat));
  ASSERT_TRUE(ph2mat.rows == 1);
  ASSERT_TRUE(ph2mat.cols == (int) PersonHistogram::MAT_COLS);
#ifndef SVM_USE_STD_DEV_HIST
  ASSERT_TRUE(ph2mat.cols == PersonHistogram::HIST_NBINS);
  ASSERT_TRUE(matrix_testing::matrices_equal(ph.get_hist_vector().at(1).t(), ph2mat));
#endif
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, merge_empty) {
  PersonHistogram ph;
  std::vector<PersonHistogram> phists;
  ASSERT_TRUE(PersonHistogram::merge_histograms(phists, ph));
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, merge_one_image) {
  PersonHistogram ph_out(alberto1_file, alberto1_seed, KINECT_SERIAL_LAB()),
      ph_in(juggling1_file, juggling1_seed, KINECT_SERIAL_LAB());
  ASSERT_SOUND_PH(ph_in, 1);
  ASSERT_SOUND_PH(ph_out, 1);
  std::vector<PersonHistogram> phists(1, ph_in);
  ASSERT_TRUE(PersonHistogram::merge_histograms(phists, ph_out));
  // check merging: now ph=ph2
  ASSERT_SOUND_PH(ph_out, 1);
  for (unsigned int hidx = 0; hidx < PersonHistogram::BODY_PARTS; ++hidx)
    ASSERT_TRUE(matrix_testing::matrices_near(ph_out.get_hist_vector()[hidx],
                                              ph_in.get_hist_vector()[hidx], 1E-2));
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, merge_two_images) {
  PersonHistogram ph_out(alberto1_file, alberto1_seed, KINECT_SERIAL_LAB()),
      ph_in1(juggling1_file, juggling1_seed, KINECT_SERIAL_LAB()),
      ph_in2(juggling2_file, juggling2_seed, KINECT_SERIAL_LAB());
  std::vector<PersonHistogram> phists;
  phists.push_back(ph_in1);
  phists.push_back(ph_in2);
  ASSERT_SOUND_PH(ph_in1, 1);
  ASSERT_SOUND_PH(ph_in2, 1);
  ASSERT_SOUND_PH(ph_out, 1);
  ASSERT_TRUE(PersonHistogram::merge_histograms(phists, ph_out));
  // check merging: now ph=ph1+ph2/2
  ASSERT_SOUND_PH(ph_out, 2);
  for (unsigned int hidx = 0; hidx < PersonHistogram::BODY_PARTS; ++hidx)
    ASSERT_TRUE(matrix_testing::matrices_near
                (ph_out.get_hist_vector()[hidx],
                 .5*(ph_in1.get_hist_vector()[hidx] + ph_in2.get_hist_vector()[hidx]),
                 1E-2));
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, merge_four_images) {
  PersonHistogram ph_out(alberto1_file, alberto1_seed, KINECT_SERIAL_LAB()),
      ph_in1(juggling1_file, juggling1_seed, KINECT_SERIAL_LAB()),
      ph_in2(juggling2_file, juggling2_seed, KINECT_SERIAL_LAB());
  std::vector<PersonHistogram> phists;
  phists.push_back(ph_in1);
  phists.push_back(ph_in2);
  phists.push_back(ph_in2);
  phists.push_back(ph_in2);
  ASSERT_TRUE(PersonHistogram::merge_histograms(phists, ph_out));
  // check merging: now ph=.25*ph1+.75*ph2
  ASSERT_SOUND_PH(ph_out, 4);
  for (unsigned int hidx = 0; hidx < PersonHistogram::BODY_PARTS; ++hidx)
    ASSERT_TRUE(matrix_testing::matrices_near
                (ph_out.get_hist_vector()[hidx],
                 .25*ph_in1.get_hist_vector()[hidx] + .75*ph_in2.get_hist_vector()[hidx],
                 1E-2));
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv){
  // Run all the tests that were declared with TEST()
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
