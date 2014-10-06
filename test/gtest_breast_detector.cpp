/*!
  \file        gtest_breast_detector.cpp
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2013/10/5

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
#include <map/map_utils.h>
#include "height_detector/breast_detector.h"
#include <vision_utils/img_path.h>
#include "image_utils/io.h"
#include "databases_io/dgaitdb_filename.h"

//#define DISPLAY

#if 0
TEST(TestSuite, test_template_fn) {
  MiniStage ms;
  ms.set_scale(1. / 500);

  std::string _window_name="foo";
  cv::namedWindow(_window_name);
  ms.set_mouse_move_callback(_window_name);
  // a: 0 -> 0.5
  // b: -0.5 -> 0.5
  // s: 0 -> 10
  int a_tb = 2, b_tb = 5, s_tb = 2;
  int tb_size = 10;
  cv::createTrackbar("a", _window_name, &a_tb, tb_size);
  cv::createTrackbar("b", _window_name, &b_tb, tb_size);
  cv::createTrackbar("s", _window_name, &s_tb, tb_size);
  BreastDetectorExtended::Template tvec;;
  BreastDetector detec;
  detec.template_matching_fn_y(tvec);
  while (true) {
    double a = a_tb * .5 / tb_size,
        b = -.5 + b_tb * 1. / tb_size,
        s = s_tb * 10 / tb_size;
    printf("loop, a:%g, b:%g, s:%g\n", a, b, s);
    ms.clear();
    ms.draw_grid(.5f, 150);
    ms.draw_axes();
    // plot shape
    BreastDetector::template_matching_fn_x(tvec, a, b, s);
    //      printf("tvec:'%s'\n", StringUtils::iterable_to_string(tvec).c_str());
    mini_stage_plugins::plot_xy(ms, tvec, CV_RGB(0,0,0), 2);
    cv::imshow(_window_name, ms.get_viz());
    char c = cv::waitKey(50);
    if ((int) c == 27)
      break;
  } // end while (true)
}
#endif

//! pass some fields and functions to a public scope
class BreastDetectorExtended : public BreastDetector {
public:
  inline void template_matching_fn(BreastDetector::Template & t) {
    BreastDetector::template_matching_fn(t);
  }
  inline std::vector<std::vector<Pt2f> > & get_slices() { return _slices; }
  inline Template* get_best_template() const { return _best_template; }
  inline bool detect_breast_template_matching_compare_slices() {
    return BreastDetector::detect_breast_template_matching_compare_slices();
  }
}; // end class BreastDetectorExtended

TEST(TestSuite, test_IndexDoubleConverter) {
  IndexDoubleConverter conv(-1, 1, 10);
  ASSERT_TRUE(conv.double2index(-1) == 0);
  ASSERT_TRUE(conv.double2index(1)  == 10);
  ASSERT_TRUE(conv.index2double(0)  == -1);
  ASSERT_TRUE(conv.index2double(10) == 1);
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, detect_breast_template_matching_compare_slices_random) {
  BreastDetectorExtended detec;
  BreastDetectorExtended::Template t(0, 0, 0);
  detec.template_matching_fn(t);
  for (unsigned int pt_idx = 0; pt_idx < t.pts.size(); ++pt_idx)
    t.pts[pt_idx].x = drand48(); // random values
  // add points to a BreastDetector slice
  detec.get_slices().push_back(t.pts);
  // now hope we match it again
  Timer timer;
  ASSERT_TRUE(detec.detect_breast_template_matching_compare_slices());
  timer.printTime("detect_breast_template_matching_compare_slices()");
}


TEST(TestSuite, detect_breast_template_matching_compare_slices_perfect_match) {
  BreastDetectorExtended detec;
  // generate a perfect breast
  BreastDetectorExtended::Template t_ref(.2, .1, 5), *t;
  detec.template_matching_fn(t_ref);
  // add points to a BreastDetector slice
  detec.get_slices().push_back(t_ref.pts);
  // now hope we match it again
  Timer timer;
  ASSERT_TRUE(detec.detect_breast_template_matching_compare_slices());
  timer.printTime("detect_breast_template_matching_compare_slices()");
  t = detec.get_best_template();
  ASSERT_NEAR(t->a, t_ref.a, 1E-1) << "best_a:" << t_ref.a;
  ASSERT_NEAR(t->b, t_ref.b, 1E-1) << "best_b:" << t_ref.b;
  ASSERT_NEAR(t->s, t_ref.s, 1E-1) << "best_s:" << t_ref.s;
}

////////////////////////////////////////////////////////////////////////////////

void load_depth_mask_cammodel(cv::Mat1f & depth,
                              cv::Mat1b & user_mask,
                              image_geometry::PinholeCameraModel & depth_camera_model,
                              const std::string filename_prefix,
                              const std::string kinect_serial_number = DEFAULT_KINECT_SERIAL()) {
  cv::Mat3b rgb;
  ASSERT_TRUE(image_utils::read_rgb_depth_user_image_from_image_file
              (filename_prefix, &rgb, &depth, &user_mask));
#ifdef DISPLAY
  cv::imshow("rgb", rgb);
  cv::imshow("depth", image_utils::depth2viz(depth));
  cv::waitKey(10);
#endif
  image_geometry::PinholeCameraModel rgb_camera_model;
  ASSERT_TRUE(kinect_openni_utils::read_camera_model_files
              (kinect_serial_number, depth_camera_model, rgb_camera_model));
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, breast_empty) {
  //BreastDetector detec(true);
  BreastDetector detec(false);
  ASSERT_NO_THROW();
}

////////////////////////////////////////////////////////////////////////////////

void test_detect_empty_args(BreastDetector::Method method) {
  cv::Mat1f depth;
  cv::Mat1b user_mask;
  image_geometry::PinholeCameraModel depth_camera_model;
  BreastDetector detec;
  BreastDetector::HeightBreast h =
      detec.detect_breast(depth, user_mask, depth_camera_model, method);
  ASSERT_NO_THROW();
  ASSERT_TRUE(h.gender == BreastDetector::ERROR); // error
}

TEST(TestSuite, detect_empty_args_walk3d) { test_detect_empty_args(BreastDetector::WALK3D); }
TEST(TestSuite, detect_empty_args_reproj) { test_detect_empty_args(BreastDetector::REPROJECT); }

////////////////////////////////////////////////////////////////////////////////

void test_detect_zero_depth(BreastDetector::Method method) {
  cv::Mat1f depth;
  cv::Mat1b user_mask;
  image_geometry::PinholeCameraModel depth_camera_model;
  load_depth_mask_cammodel(depth, user_mask, depth_camera_model, IMG_DIR "depth/juggling1");
  depth.setTo(0);
  BreastDetector detec;
  BreastDetector::HeightBreast h =
      detec.detect_breast(depth, user_mask, depth_camera_model, method);
  ASSERT_TRUE(h.gender == BreastDetector::ERROR); // error
}

TEST(TestSuite, detect_zero_depth_walk3d) { test_detect_zero_depth(BreastDetector::WALK3D); }
TEST(TestSuite, detect_zero_depth_reproj) { test_detect_zero_depth(BreastDetector::REPROJECT); }

////////////////////////////////////////////////////////////////////////////////

void test_mask(BreastDetector & detec, uchar mask_value, BreastDetector::Method method) {
  cv::Mat1f depth;
  cv::Mat1b user_mask;
  image_geometry::PinholeCameraModel depth_camera_model;
  load_depth_mask_cammodel(depth, user_mask, depth_camera_model, IMG_DIR "depth/juggling1");
  user_mask.setTo(mask_value);
  BreastDetector::HeightBreast h =
      detec.detect_breast(depth, user_mask, depth_camera_model, method);
  if (mask_value) {
    ASSERT_TRUE(h.gender >= 0); // success
    //detec.breast2img(user_mask, 10);
  }
  else
    ASSERT_TRUE(h.gender == BreastDetector::ERROR); // error
} // end test_mask();

TEST(TestSuite, height_pixels_zero_mask) {
  BreastDetector detec;
  test_mask(detec, 0, BreastDetector::WALK3D);
  // test_mask(detec, 0, BreastDetector::REPROJECT);
  test_mask(detec, 0, BreastDetector::TEMPLATE_MATCHING);
}

void test_height_pixels_full_mask(BreastDetector & detec) {
  maggiePrint("test_height_pixels_full_mask()");

  test_mask(detec, 255, BreastDetector::WALK3D);
  // test_mask(detec, 255, BreastDetector::REPROJECT);
  test_mask(detec, 255, BreastDetector::TEMPLATE_MATCHING);
}

////////////////////////////////////////////////////////////////////////////////

void test_breast(BreastDetector & detec,
                 const std::string filename_prefix,
                 uchar user_idx,
                 const std::string kinect_serial_number,
                 BreastDetector::Method method,
                 bool must_pass = true) {
  cv::Mat1f depth;
  cv::Mat1b user_mask;
  image_geometry::PinholeCameraModel depth_camera_model;
  load_depth_mask_cammodel(depth, user_mask, depth_camera_model,
                           filename_prefix, kinect_serial_number);
  cv::Mat1b curr_user_mask = (user_mask == user_idx);
  Timer timer;
  BreastDetector::HeightBreast h =
      detec.detect_breast(depth, curr_user_mask, depth_camera_model, method);
  printf("Time for detect_breast(method %i) : %g ms\n", method, timer.getTimeMilliseconds());
  if (must_pass)
    ASSERT_TRUE(h.gender >= 0); // success
  else
    ASSERT_TRUE(h.gender == BreastDetector::ERROR); // failure

#ifdef DISPLAY
  cv::Mat3b breast_illus(user_mask.size(), cv::Vec3b(0,0,0));
  detec.breast2img(user_mask, breast_illus, h, method);
  detec.illus_pcl(depth, user_mask, depth_camera_model, method);
  cv::imshow("breast_illus", breast_illus); cv::waitKey(0); // cv::destroyAllWindows();
#endif
}

void test_breast(BreastDetector & detec,
                 const std::string filename_prefix,
                 uchar user_idx,
                 const std::string kinect_serial_number = DEFAULT_KINECT_SERIAL(),
                 bool must_pass = true) {
  printf("test_breast('%s')\n", filename_prefix.c_str());
  test_breast(detec, filename_prefix, user_idx, kinect_serial_number, BreastDetector::WALK3D, must_pass);
  //test_breast(detec, filename_prefix, user_idx, kinect_serial_number, BreastDetector::REPROJECT, must_pass);
  //test_breast(detec, filename_prefix, user_idx, kinect_serial_number, BreastDetector::TEMPLATE_MATCHING, must_pass);
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, breast_empty_lab) {
  BreastDetector detec;
  test_breast(detec, IMG_DIR "depth/empty_lab", 1, KINECT_SERIAL_LAB(), false);
}

TEST(TestSuite, breast_wrong_user_idx) {
  BreastDetector detec;
  test_breast(detec, IMG_DIR "depth/david_arnaud2", 10, KINECT_SERIAL_LAB(), false);
}

void test_breast_DGaitDB(BreastDetector & detec) {
  maggiePrint("test_breast_DGaitDB()");

  DGaitDBFilename f("/home/user/Downloads/0datasets/DGaitDB_imgs/");
  if (!f.directory_exists())
    return;
  for (unsigned int oni_idx = 1; oni_idx <= DGaitDBFilename::ONI_FILES; ++oni_idx) {
    test_breast(detec, f.filename_train(oni_idx, 1), DGaitDBFilename::USER_IDX, DEFAULT_KINECT_SERIAL());
    //for (unsigned int train_idx = 1; train_idx <= DGaitDBFilename::NFILES_TRAIN; ++train_idx)
    //  test_breast(detec, f.filename_train(oni_idx, train_idx), DGaitDBFilename::USER_IDX, DEFAULT_KINECT_SERIAL());
  } // end loop oni_idx
}

void test_breast_ainara(BreastDetector & detec) {
  maggiePrint("test_breast_ainara()");

  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-13-769", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-17-635", 4, KINECT_SERIAL_ARNAUD());
  //
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-03-286", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-03-861", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-05-027", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-14-841", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-16-198", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-18-208", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-18-804", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-22-545", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-23-350", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-24-254", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-26-315", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-27-255", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-29-673", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-32-351", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-35-989", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-40-027", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-41-242", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-45-296", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-45-998", 4, KINECT_SERIAL_ARNAUD());
  test_breast(detec, IMG_DIR "breast/2013-10-05_15-46-46-901", 4, KINECT_SERIAL_ARNAUD());
}

void test_breast_men(BreastDetector & detec) {
  maggiePrint("test_breast_men()");

  test_breast(detec, IMG_DIR "depth/alberto1", 255, KINECT_SERIAL_LAB());
  test_breast(detec, IMG_DIR "depth/david_arnaud1", 1, KINECT_SERIAL_LAB());
  test_breast(detec, IMG_DIR "depth/david_arnaud1", 2, KINECT_SERIAL_LAB());
  test_breast(detec, IMG_DIR "depth/david_arnaud2", 1, KINECT_SERIAL_LAB());
  test_breast(detec, IMG_DIR "depth/david_arnaud2", 2, KINECT_SERIAL_LAB());
  test_breast(detec, IMG_DIR "depth/david_arnaud3", 1, KINECT_SERIAL_LAB());
  test_breast(detec, IMG_DIR "depth/david_arnaud3", 2, KINECT_SERIAL_LAB());
  test_breast(detec, IMG_DIR "depth/juggling1", 255, KINECT_SERIAL_LAB());
  test_breast(detec, IMG_DIR "depth/juggling2", 255, KINECT_SERIAL_LAB());
}

////////////////////////////////////////////////////////////////////////////////

void test_breast_all_values(BreastDetector & detec,
                            const std::string filename_prefix,
                            unsigned int nusers,
                            const std::string kinect_serial_number,
                            BreastDetector::Method method) {
  cv::Mat1f depth;
  cv::Mat1b user_mask;
  image_geometry::PinholeCameraModel depth_camera_model;
  load_depth_mask_cammodel(depth, user_mask, depth_camera_model,
                           filename_prefix, kinect_serial_number);
  std::map<int, BreastDetector::HeightBreast> heights;
  cv::Mat3b breasts_illus;
  bool ok = detec.breast_all_values(depth, user_mask, depth_camera_model,
                                    heights, method, true, &breasts_illus);
  ASSERT_TRUE(ok); // success
  ASSERT_TRUE(heights.size() == nusers); // success
  for(std::map<int, BreastDetector::HeightBreast>::const_iterator hit = heights.begin();
      hit != heights.end(); ++hit) {
    ASSERT_TRUE(hit->second.gender >= 0); // success
  }
#ifdef DISPLAY
  cv::imshow("breasts_illus", breasts_illus); cv::waitKey(0); cv::destroyAllWindows();
#endif
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, breast_all_values_empty_lab) {
  BreastDetector detec;
  test_breast_all_values(detec, IMG_DIR "depth/empty_lab", 0,
                         KINECT_SERIAL_LAB(), BreastDetector::WALK3D);
  test_breast_all_values(detec, IMG_DIR "depth/empty_lab", 0,
                         KINECT_SERIAL_LAB(), BreastDetector::REPROJECT);
  test_breast_all_values(detec, IMG_DIR "depth/empty_lab", 0,
                         KINECT_SERIAL_LAB(), BreastDetector::TEMPLATE_MATCHING);
}

void test_breast_all_values_ainara(BreastDetector & detec) {
  maggiePrint("test_breast_all_values_ainara()");

  test_breast_all_values(detec, IMG_DIR "breast/2013-10-05_15-46-13-769", 1,
                         KINECT_SERIAL_ARNAUD(), BreastDetector::WALK3D);
  //  test_breast_all_values(detec, IMG_DIR "breast/2013-10-05_15-46-13-769", 1,
  //                         KINECT_SERIAL_ARNAUD(), BreastDetector::REPROJECT);
  test_breast_all_values(detec, IMG_DIR "breast/2013-10-05_15-46-13-769", 1,
                         KINECT_SERIAL_ARNAUD(), BreastDetector::TEMPLATE_MATCHING);
}

void test_breast_all_values_men(BreastDetector & detec) {
  maggiePrint("test_breast_all_values_men()");

  test_breast_all_values(detec, IMG_DIR "depth/david_arnaud2", 2,
                         KINECT_SERIAL_LAB(), BreastDetector::WALK3D);
  //  test_breast_all_values(detec, IMG_DIR "depth/david_arnaud2", 2,
  //                         KINECT_SERIAL_LAB(), BreastDetector::REPROJECT);
  test_breast_all_values(detec, IMG_DIR "depth/david_arnaud2", 2,
                         KINECT_SERIAL_LAB(), BreastDetector::TEMPLATE_MATCHING);
}

TEST(TestSuite, big_test) {
  BreastDetector detec;
  test_height_pixels_full_mask(detec);
  test_breast_DGaitDB(detec);
  test_breast_ainara(detec);
  test_breast_men(detec);
  test_breast_all_values_ainara(detec);
  test_breast_all_values_men(detec);
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv){
  // Run all the tests that were declared with TEST()

  // octave test -
  //http://www.gnu.org/software/octave/doc/interpreter/Standalone-Programs.html#Standalone-Programs
  // http://en.wikipedia.org/wiki/GNU_Octave#C.2B.2B_integration

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

