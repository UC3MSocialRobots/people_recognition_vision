/*!
  \file        gtest_height_detector.cpp
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2013/9/24

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

Some tests for HeightDetector

 */

// Bring in gtest
#include <gtest/gtest.h>
#include <map/map_utils.h>
#include <time/timer.h>
#include "people_recognition_vision/height_detector.h"

//#define DISPLAY

void load_depth_mask_cammodel(cv::Mat1f & depth,
                              cv::Mat1b & user_mask,
                              image_geometry::PinholeCameraModel & depth_camera_model,
                              const std::string filename_prefix,
                              const std::string kinect_serial_number = DEFAULT_KINECT_SERIAL()) {
  image_utils::read_rgb_depth_user_image_from_image_file
      (filename_prefix, NULL, &depth, &user_mask);
  image_geometry::PinholeCameraModel rgb_camera_model;
  ASSERT_TRUE(kinect_openni_utils::read_camera_model_files
              (kinect_serial_number, depth_camera_model, rgb_camera_model));
}

////////////////////////////////////////////////////////////////////////////////
/// height_pixels()
////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, height_pixels_empty_img) {
  cv::Mat1b query;
  HeightDetector detec;
  HeightDetector::Height h = detec.height_pixels(query);
  ASSERT_TRUE(h.height_px == HeightDetector::ERROR)
      << "query:" << ImageContour::to_string(query)
      << "height:" << h.to_string();
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, height_pixels_black_img) {
  int cols = 100;
  cv::Mat1b query(cols, cols, (uchar) 0);
  HeightDetector detec;
  HeightDetector::Height h = detec.height_pixels(query);
  ASSERT_TRUE(h.height_px == HeightDetector::ERROR)
      << "query:" << ImageContour::to_string(query)
      << "height:" << h.to_string();
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, height_pixels_white_img) {
  int cols = 100;
  cv::Mat1b query(cols, cols, (uchar) 255);
  HeightDetector detec;
  HeightDetector::Height h = detec.height_pixels(query);
  ASSERT_TRUE(h.height_px == cols) << "query:" << ImageContour::to_string(query)
                                   << "height:" << h.to_string();
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, height_pixels_single_pt) {
  int cols = 100;
  cv::Mat1b query(cols, cols, (uchar) 0);
  query(cols / 2, cols / 2) = 255;
  HeightDetector detec;
  HeightDetector::Height h = detec.height_pixels(query);
  ASSERT_TRUE(h.height_px == 1) << "query:" << ImageContour::to_string(query)
                                << "height:" << h.to_string();
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, height_pixels_circle) {
  unsigned int ntimes = 10;
  for (unsigned int time = 0; time < ntimes; ++time) {
    printf("iter\n\n");
    int cols = 4 * (2 + (rand() % 10));
    cv::Mat1b query(cols, cols, (uchar) 0);
    cv::circle(query, cv::Point(cols / 2, cols / 2), cols / 4, cv::Scalar::all(255), -1);
    HeightDetector detec;
    HeightDetector::Height h = detec.height_pixels(query);
    ASSERT_TRUE(h.height_px == 1 + cols / 2) << "query:" << ImageContour::to_string(query)
                                             << "height:" << h.to_string()
                                             << ", exp height:" << 1 + cols / 2;
  } // end loop time
}

////////////////////////////////////////////////////////////////////////////////

void test_height_pixels(const std::string user_mask_name,
                        int exp_height_pixels, int height_error) {
  cv::Mat1b user_mask = cv::imread(user_mask_name, CV_LOAD_IMAGE_GRAYSCALE);
  HeightDetector detector;
  HeightDetector::Height h;
  unsigned int ntimes = 10;
  Timer timer;
  for (unsigned int time = 0; time < ntimes; ++time)
    h = detector.height_pixels(user_mask);
  timer.printTime_factor("height_pixels()", ntimes);
  std::cout << user_mask_name << ": time:" << timer.getTimeMilliseconds()/ ntimes
            << ", height:" << h.to_string() << std::endl;

  // display
  timer.reset();
  cv::Mat3b height2img_illus;
  for (unsigned int time = 0; time < ntimes; ++time)
    detector.height2img(user_mask, height2img_illus, true, h);
  timer.printTime_factor("height2img()", ntimes);
#ifdef DISPLAY
  cv::imshow("height2img_illus", height2img_illus); cv::waitKey(0);
#endif // DISPLAY

  // check results
  ASSERT_TRUE(h.height_px != HeightDetector::NOT_COMPUTED);
  ASSERT_TRUE(h.height_px != HeightDetector::ERROR);
  ASSERT_NEAR(h.height_px, exp_height_pixels, height_error)
      << "height:" << h.to_string()
      << "exp_height_pixels:" << exp_height_pixels;
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, height_pixels_heads_01) {
  test_height_pixels(IMG_DIR "skeletons/heads/01.png", 350, 50);
}
TEST(TestSuite, height_pixels_heads_02) {
  test_height_pixels(IMG_DIR "skeletons/heads/02.png", 350, 50);
}
TEST(TestSuite, height_pixels_heads_03) {
  test_height_pixels(IMG_DIR "skeletons/heads/03.png", 350, 50);
}
TEST(TestSuite, height_pixels_heads_04) {
  test_height_pixels(IMG_DIR "skeletons/heads/04.png", 350, 50);
}
TEST(TestSuite, height_pixels_heads_05) {
  test_height_pixels(IMG_DIR "skeletons/heads/05.png", 400, 50);
}
TEST(TestSuite, height_pixels_heads_06) {
  test_height_pixels(IMG_DIR "skeletons/heads/06.png", 370, 50);
}
TEST(TestSuite, height_pixels_hard_mask1) {
  test_height_pixels(IMG_DIR "skeletons/heads/hard_mask1.png", 240, 50);
}
TEST(TestSuite, height_pixels_hard_mask2) {
  test_height_pixels(IMG_DIR "skeletons/heads/hard_mask2.png", 274, 50);
}
TEST(TestSuite, height_pixels_hard_mask3) {
  test_height_pixels(IMG_DIR "skeletons/heads/hard_mask3.png", 274, 50);
}

////////////////////////////////////////////////////////////////////////////////
/// height_meters()
////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, height_meters_empty) {
  cv::Mat1f depth;
  cv::Mat1b user_mask;
  image_geometry::PinholeCameraModel depth_camera_model;
  HeightDetector detec;
  HeightDetector::Height h = detec.height_meters(depth, user_mask, depth_camera_model);
  ASSERT_TRUE(h.height_m == HeightDetector::ERROR)
      << "user_mask:" << ImageContour::to_string(user_mask)
      << "height:" << h.to_string();
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, height_meters_zero_mask) {
  cv::Mat1f depth;
  cv::Mat1b user_mask;
  image_geometry::PinholeCameraModel depth_camera_model;
  load_depth_mask_cammodel(depth, user_mask, depth_camera_model, IMG_DIR "depth/juggling1");
  user_mask.setTo(0);
  HeightDetector detec;
  HeightDetector::Height h = detec.height_meters(depth, user_mask, depth_camera_model);
  ASSERT_TRUE(h.height_m == HeightDetector::ERROR)
      << "user_mask:" << ImageContour::to_string(user_mask)
      << "height:" << h.to_string();
}

////////////////////////////////////////////////////////////////////////////////

void test_height_meters(std::string prefix,
                        double exp_height_m, double height_error,
                        bool compute_confidence = false,
                        double exp_confidence = -1, double lkl_error = -1) {
  cv::Mat1f depth;
  cv::Mat1b user_mask;
  image_geometry::PinholeCameraModel depth_camera_model;
  load_depth_mask_cammodel(depth, user_mask, depth_camera_model, prefix);
  HeightDetector detec;

  Timer timer;
  HeightDetector::Height h = detec.height_meters
                             (depth, user_mask, depth_camera_model, compute_confidence);
  timer.printTime("height_meters()");
  std::cout << prefix << ":height:" << h.to_string() << std::endl;
  ASSERT_TRUE(h.height_m != HeightDetector::NOT_COMPUTED);
  ASSERT_TRUE(h.height_m != HeightDetector::ERROR);
  ASSERT_NEAR(h.height_m, exp_height_m, height_error)
      << "user_mask:" << ImageContour::to_string(user_mask)
      << "height:" << h.to_string()
      << "exp_height_m:" << exp_height_m;

  // viz
#ifdef DISPLAY
  cv::Mat3b viz_img(user_mask.size());
  detec.height2img(user_mask, viz_img, true, h);
  timer.printTime("height2img()");
  cv::imshow("viz_img", viz_img); cv::waitKey(0);
#endif

  if (!compute_confidence)
    return;
  ASSERT_TRUE(h.height_confidence != HeightDetector::NOT_COMPUTED);
  ASSERT_TRUE(h.height_confidence != HeightDetector::ERROR);
  ASSERT_NEAR(h.height_confidence, exp_confidence, lkl_error)
      << "user_mask:" << ImageContour::to_string(user_mask)
      << "height:" << h.to_string()
      << "exp_confidence:" << exp_confidence;
}

TEST(TestSuite, height_meters_juggling1) {
  test_height_meters(IMG_DIR "depth/juggling1", 1.8, .2, false);
}

TEST(TestSuite, height_meters_juggling2) {
  test_height_meters(IMG_DIR "depth/juggling2", 1.8, .2, false);
}

TEST(TestSuite, height_meters_alberto1) {
  test_height_meters(IMG_DIR "depth/alberto1", 1.8, .2, false);
}

////////////////////////////////////////////////////////////////////////////////
/// height_meters_all_values()
////////////////////////////////////////////////////////////////////////////////

void test_height_all_values(const std::string filename_prefix,
                            unsigned int exp_nusers,
                            const std::string kinect_serial_number = DEFAULT_KINECT_SERIAL()) {
  cv::Mat1f depth;
  cv::Mat1b user_mask;
  image_geometry::PinholeCameraModel depth_camera_model;
  load_depth_mask_cammodel(depth, user_mask, depth_camera_model, filename_prefix, kinect_serial_number);

  HeightDetector detec;
  std::map<int, HeightDetector::Height> heights;
  cv::Mat3b illus;
  Timer timer;
  bool ok = detec.height_meters_all_values
            (depth, user_mask, depth_camera_model, heights, true, true, &illus);
  timer.printTime("height_meters_all_values()");
  ASSERT_TRUE(ok);
#ifdef DISPLAY
  cv::imshow("illus", illus); cv::waitKey(0);
#endif

#if 0
  std::vector<int> out_keys, correct_keys;
  for (unsigned int i = 0; i < exp_nusers; ++i)
    correct_keys.push_back(i+1);
  map_utils::map_keys_to_container(heights, out_keys);
  ASSERT_TRUE(out_keys == correct_keys)
      << "heights:" << StringUtils::map_to_string(heights);
#else
  ASSERT_TRUE(heights.size() == exp_nusers)
      << "exp_nusers:" << exp_nusers
      << ", heights:" << StringUtils::map_to_string(heights);
#endif
  std::map<int, HeightDetector::Height>::const_iterator h_it = heights.begin();
  while(h_it != heights.end()) {
    ASSERT_TRUE(h_it->second.height_px != HeightDetector::NOT_COMPUTED);
    ASSERT_TRUE(h_it->second.height_px != HeightDetector::ERROR);
    ASSERT_TRUE(h_it->second.height_confidence != HeightDetector::NOT_COMPUTED);
    ASSERT_TRUE(h_it->second.height_confidence != HeightDetector::ERROR);
    ++h_it;
  } // end for (h_it)
} // end test_height_all_values();

TEST(TestSuite, height_all_values_empty_lab) {
  test_height_all_values(IMG_DIR "depth/empty_lab", 0, KINECT_SERIAL_LAB());
}

TEST(TestSuite, height_all_values_juggling1) {
  test_height_all_values(IMG_DIR "depth/juggling1", 1, KINECT_SERIAL_LAB());
}

TEST(TestSuite, height_all_values_david_arnaud1) {
  test_height_all_values(IMG_DIR "depth/david_arnaud1", 2, KINECT_SERIAL_LAB());
}

TEST(TestSuite, height_all_values_david_arnaud2) {
  test_height_all_values(IMG_DIR "depth/david_arnaud2", 2, KINECT_SERIAL_LAB());
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, ref_skel) {
  HeightDetector detec;
  cv::Mat1b ref_skel = cv::imread(detec.get_ref_skel_filename(), CV_LOAD_IMAGE_GRAYSCALE);
  ref_skel = (ref_skel == 255);
  // cv::imshow("ref_skel", ref_skel); cv::waitKey(0);
  HeightDetector::Height h = detec.height_pixels(ref_skel, true);
  ASSERT_TRUE(h.height_px != HeightDetector::NOT_COMPUTED);
  ASSERT_TRUE(h.height_px != HeightDetector::ERROR);
  ASSERT_TRUE(h.height_confidence != HeightDetector::NOT_COMPUTED);
  ASSERT_TRUE(h.height_confidence != HeightDetector::ERROR);
  ASSERT_NEAR(h.height_confidence, 1, .1)
      // << "user_mask:" << ImageContour::to_string(user_mask)
      << "height:" << h.to_string();
}

////////////////////////////////////////////////////////////////////////////////
#include <databases_io/g3d2imgs.h>
#define G3D_DIR "/home/user/Downloads/0datasets/g3d_kingston/"
TEST(TestSuite, g3d2imgs) {
  G3D2Imgs db;
  if (!db.from_file(G3D_DIR "Fighting/KinectOutput22/Colour/Colour 12.png"))
    return;
  ASSERT_TRUE(db.go_to_next_frame());
  image_geometry::PinholeCameraModel depth_camera_model, rgb_camera_model;
  ASSERT_TRUE(kinect_openni_utils::read_camera_model_files
              (DEFAULT_KINECT_SERIAL(), depth_camera_model, rgb_camera_model));

  std::map<int, HeightDetector::Height> heights;
  cv::Mat3b illus;
  Timer timer;
  HeightDetector detec;
  bool ok = detec.height_meters_all_values
            (db.get_depth(), db.get_user(), depth_camera_model, heights, true, true, &illus);
  timer.printTime("height_meters_all_values()");
  printf("heights:'%s'\n", StringUtils::map_to_string(heights).c_str());
  ASSERT_TRUE(ok);
#ifdef DISPLAY
  db.display();
  cv::imshow("illus", illus); cv::waitKey(0);
#endif
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv){
  // Run all the tests that were declared with TEST()
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
