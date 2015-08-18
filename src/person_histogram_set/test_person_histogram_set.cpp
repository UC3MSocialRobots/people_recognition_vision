/*!
  \file        test_person_histogram_set.cpp
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2012/12/24

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
// OpenCV
#include <opencv2/highgui/highgui.hpp>
// AD
#include "kinect/user_image_to_rgb.h"
#include "time/timer.h"
#include "vision_utils/test_person_histogram_set_variables.h"
#include "person_histogram_set.h"

////////////////////////////////////////////////////////////////////////////////

inline void test_depth_canny(const std::string filename_prefix) {
  cv::Mat rgb, depth;
  image_utils::read_rgb_and_depth_image_from_image_file(filename_prefix, &rgb, &depth);
  // image_utils::read_depth_rgb_from_yaml_file(filename_prefix, rgb, depth);

  DepthCanny _depth_canny;
  Timer timer;
  unsigned int ntimes = 1;
  for (unsigned int i = 0; i < ntimes; ++i)
    _depth_canny.thresh(depth);
  timer.printTime_factor("thres()", ntimes);

  TIMER_DISPLAY_CHART(_depth_canny.timer, 1);
  cv::imshow("rgb", rgb);
  cv::imshow("depth_illus", image_utils::depth2viz(depth, image_utils::FULL_RGB_STRETCHED));
  cv::imshow("img_uchar_with_no_nan", _depth_canny._img_uchar_with_no_nan);
  cv::imshow("edges", _depth_canny._edges);
  cv::imshow("thresholded_image", _depth_canny.get_thresholded_image());
  // image_utils::imwrite_debug("broken_edge.png", dit.get_thresholded_image());
  cv::waitKey(0); cv::destroyAllWindows();
} // end test_depth_canny();

////////////////////////////////////////////////////////////////////////////////
#if 0

inline void test_person_compute_user_mask(const std::string filename_prefix,
                                          const cv::Point & seed) {
  cv::Mat rgb, depth;
  image_utils::read_rgb_and_depth_image_from_image_file(filename_prefix, &rgb, &depth);
  // image_utils::read_depth_rgb_from_yaml_file(filename_prefix, rgb, depth);

  PersonHistogram ph;
  Timer timer;
  unsigned int n_times  = 100;
  for (unsigned int i = 0; i < n_times; ++i)
    ph.compute_user_mask(depth, seed);
  timer.printTime_factor("PersonHistogram::compute_user_mask()", n_times);

  TIMER_DISPLAY_CHART(ph._depth_canny.timer, 1);
  cv::imshow("rgb", rgb);
  cv::imshow("depth_illus", image_utils::depth2viz(depth, image_utils::FULL_RGB_STRETCHED));
  cv::imshow("depth_canny", ph._depth_canny.get_thresholded_image());
  cv::imshow("user_mask", ph.get_user_mask());
  //  image_utils::imwrite_debug("mask.png", ph.get_user_mask());
  //  image_utils::imwrite_debug("rgb.png", rgb);
  cv::waitKey(0); cv::destroyAllWindows();
}

////////////////////////////////////////////////////////////////////////////////

inline void test_find_mask_then_top_point_centered(const std::string filename_prefix,
                                                   const cv::Point & seed) {
  cv::Mat rgb, depth;
  image_utils::read_rgb_and_depth_image_from_image_file(filename_prefix, &rgb, &depth);
  PersonHistogram ph;
  ph.compute_user_mask(depth, seed);
  //  find_top_point_centered
  cv::Rect search_window;
  cv::Point top_point = image_utils::find_top_point_centered
                        (ph.get_user_mask(), true, .4, (uchar) 0, &search_window);

  cv::Mat3b mask_illus;
  cv::cvtColor(ph.get_user_mask(), mask_illus, CV_GRAY2BGR);
  cv::rectangle(mask_illus, search_window, CV_RGB(255, 0, 0));
  cv::circle(mask_illus, top_point, 3, CV_RGB(0, 0, 255), 2);

  TIMER_DISPLAY_CHART(ph._depth_canny.timer, 1);
  cv::imshow("ph.get_user_mask()", ph.get_user_mask());
  cv::imshow("mask_illus", mask_illus);
  cv::waitKey(0); cv::destroyAllWindows();
}

#endif

////////////////////////////////////////////////////////////////////////////////

inline void convert_yaml_to_images(std::string filename_prefix) {
  cv::Mat rgb, depth;
  image_utils::read_depth_rgb_from_yaml_file(filename_prefix, rgb, depth);
  image_utils::write_rgb_and_depth_image_to_image_file(filename_prefix, &rgb, &depth);
}

////////////////////////////////////////////////////////////////////////////////

inline void test_person_vector_of_histograms(const std::string & filename_prefix,
                                             const cv::Point & seed,
                                             const std::string & kinect_serial_number) {
  cv::Mat rgb, depth;
  image_utils::read_rgb_and_depth_image_from_image_file(filename_prefix, &rgb, &depth);
  // image_utils::read_depth_rgb_from_yaml_file(filename_prefix, rgb, depth);

  PersonHistogram ph;
  Timer timer;
  unsigned int n_times  = 1;
  for (unsigned int i = 0; i < n_times; ++i)
    ph.create(rgb, depth, seed, kinect_serial_number,
              (i == 0) // illus images only first time
              );
  timer.printTime_factor("PersonHistogram::compute_user_mask_then_vector_of_histograms()", n_times);

  // paint results
  cv::Mat3b hist_illus;
  histogram_utils::vector_of_histograms_to_image
      (ph.get_hist_vector(), hist_illus, 200, 200, colormaps::ratio2hue);
  cv::Mat3b multimask_illus;
  user_image_to_rgb(ph.get_multimask(), multimask_illus, 8);
  TIMER_DISPLAY_CHART(ph._depth_canny.timer, 1);

  //image_utils::imwrite_debug("rgb.png", rgb);
  //image_utils::imwrite_debug("depth_illus.png", image_utils::depth2viz(depth, image_utils::FULL_RGB_STRETCHED), image_utils::COLORS256);
  image_utils::imwrite_debug("illus_color_img.png", ph.get_illus_color_img());
  image_utils::imwrite_debug("illus_color_mask.png", ph.get_illus_color_mask(), image_utils::MONOCHROME);
  image_utils::imwrite_debug("seen_buffer2img.png", ph.seen_buffer2img(), image_utils::COLORS256);
  image_utils::imwrite_debug("multimask_illus.png", multimask_illus, image_utils::COLORS256);
  image_utils::imwrite_debug("hist_illus.png", hist_illus, image_utils::COLORS256);

  cv::imshow("rgb", rgb);
  cv::imshow("depth_illus", image_utils::depth2viz(depth, image_utils::FULL_RGB_STRETCHED));
  cv::imshow("illus_color_img", ph.get_illus_color_img());
  cv::imshow("illus_color_mask", ph.get_illus_color_mask());
  cv::imshow("seen_buffer2img", ph.seen_buffer2img());
  cv::imshow("multimask_illus", multimask_illus);
  cv::imshow("hist_illus", hist_illus);
  cv::waitKey(0); cv::destroyAllWindows();
} // end test_person_vector_of_histograms();

////////////////////////////////////////////////////////////////////////////////

inline void test_factory_from_vector(const std::vector<std::string> & filename_prefixes,
                                     const std::vector<cv::Point> & seeds,
                                     const std::vector<std::string> & kinect_serial_numbers) {
  PersonHistogram ph(filename_prefixes, seeds, kinect_serial_numbers);
  unsigned int nph = filename_prefixes.size();
  int hist_w = 200, hist_h = 200;
  // paint results
  std::vector<cv::Mat3b> each_hist_illus;
  cv::Mat3b hist_illus_buffer;
  histogram_utils::vector_of_histograms_to_image
      (ph.get_hist_vector(), hist_illus_buffer, hist_w, hist_h, colormaps::ratio2hue);
  each_hist_illus.push_back(hist_illus_buffer.clone());

  printf("paint the histogram of each input...\n");
  for (unsigned int ph_idx = 0; ph_idx < nph; ++ph_idx) {
    PersonHistogram ph2(filename_prefixes[ph_idx], seeds[ph_idx], kinect_serial_numbers[ph_idx]);
    histogram_utils::vector_of_histograms_to_image
        (ph2.get_hist_vector(), hist_illus_buffer, hist_w, hist_h, colormaps::ratio2hue);
    each_hist_illus.push_back(hist_illus_buffer.clone());
  } // end loop ph_idx
  image_utils::paste_images(each_hist_illus, hist_illus_buffer,
                            true, hist_w, 3 * hist_h, 2, false);

  cv::imshow("hist_illus_buffer", hist_illus_buffer);
  cv::imshow("ph.get_illus_color_img()", ph.get_illus_color_img());
  cv::waitKey(0); cv::destroyAllWindows();
} // end test_factory_from_vector();

////////////////////////////////////////////////////////////////////////////////

inline void test_multi_hist(const std::vector<std::string> & filename_prefixes,
                            const std::vector<cv::Point> & seeds,
                            const std::vector<std::string> & kinect_serial_numbers,
                            const std::vector<PersonHistogramSet::PersonLabel> & labels) {
  // compute all histogams
  Timer timer;
  PersonHistogramSet phset;
  phset.push_back_vec(filename_prefixes, seeds, kinect_serial_numbers, labels);
  timer.printTime("loading data and computing histograms");

  // compare all histograms
  timer.reset();
  PersonHistogramSet::PHMatch assign;
  phset.compare_to(phset, assign, true, true, CV_COMP_BHATTACHARYYA, true);
  timer.printTime("compare_to()");
  maggiePrint("assign:%s, dist_matrix:'%s'",
              assignment_utils::assignment_list_to_string(assign).c_str(),
              phset.get_dist_matrix().to_string(15).c_str());

  TIMER_DISPLAY_CHART(phset.front()._depth_canny.timer, 1);
  cv::imshow("dist_matrix_illus", phset.get_dist_matrix_illus());
  cv::imshow("dist_matrix_colormap_caption", phset.get_dist_matrix_colormap_caption());
  cv::imshow("dist_matrix_illus_caption", phset.get_dist_matrix_illus_caption1());
  cv::waitKey(0);


  image_utils::imwrite_debug("/tmp/dist_matrix_illus.png", phset.get_dist_matrix_illus());
  image_utils::imwrite_debug("/tmp/dist_matrix_colormap_caption.png", phset.get_dist_matrix_colormap_caption());
  image_utils::imwrite_debug("/tmp/dist_matrix_illus_caption.png", phset.get_dist_matrix_illus_caption1());
}

#include "vision_utils/io.h"

////////////////////////////////////////////////////////////////////////////////

inline void test_structured_person_histogram_set() {
  Timer timer;
  PersonHistogramSet phset;
  phset.push_back_vec(test_person_histogram_set_variables::all_filename_prefixes_struct(),
                      test_person_histogram_set_variables::all_seeds_struct(),
                      test_person_histogram_set_variables::all_kinect_serials_struct(),
                      test_person_histogram_set_variables::all_labels_struct(),
                      true, true);
  timer.printTime("PersonHistogramSet ctor");

  timer.reset();
  phset.to_yaml("/tmp/FooPersonHistogramSet.yaml");
  timer.printTime("to yaml");
  PersonHistogramSet phset2;
  phset2.from_yaml("/tmp/FooPersonHistogramSet.yaml");
  timer.printTime("to yaml -> from yaml");

  PersonHistogramSet::PHMatch assign;
  phset.compare_to(phset, assign, true, true, CV_COMP_BHATTACHARYYA, true);
  maggiePrint("assign:%s, dist_matrix:'%s'",
              assignment_utils::assignment_list_to_string(assign).c_str(),
              phset.get_dist_matrix().to_string(15).c_str());

  TIMER_DISPLAY_CHART(phset.front()._depth_canny.timer, 1);
  cv::imshow("dist_matrix_illus", phset.get_dist_matrix_illus());
  cv::imshow("dist_matrix_colormap_caption", phset.get_dist_matrix_colormap_caption());
  cv::imshow("dist_matrix_illus_caption", phset.get_dist_matrix_illus_caption1());
  cv::waitKey(0);
}

////////////////////////////////////////////////////////////////////////////////

inline void test_structured_person_histogram_set_reco() {
  Timer timer;
  PersonHistogramSet phset, phset2;
  phset.push_back_vec(test_person_histogram_set_variables::all_filename_prefixes_struct(),
                      test_person_histogram_set_variables::all_seeds_struct(),
                      test_person_histogram_set_variables::all_kinect_serials_struct(),
                      test_person_histogram_set_variables::all_labels_struct());
  //  phset2.create(test_person_histogram_set_variables::all_filename_prefixes,
  //                test_person_histogram_set_variables::all_seeds,
  //                test_person_histogram_set_variables::all_kinect_serial_numbers);
  PersonHistogram ph(test_person_histogram_set_variables::alvaro1_file,
                     test_person_histogram_set_variables::alvaro1_seed,
                     KINECT_SERIAL_LAB());
  //~ PersonHistogram ph(test_person_histogram_set_variables::juggling1_file,
  //~ test_person_histogram_set_variables::juggling1_pt,
  //~ KINECT_SERIAL_LAB());
  timer.printTime("PersonHistogramSet ctors");

  int best_idx;
  // phset.compare_to(phset2, best_idx, true, true, CV_COMP_BHATTACHARYYA, false);
  phset.compare_to(ph, best_idx, true, true, CV_COMP_BHATTACHARYYA);
  maggiePrint("best_idx:%i, dist_matrix:'%s'",
              best_idx, phset.get_dist_matrix().to_string(15).c_str());

  TIMER_DISPLAY_CHART(phset.front()._depth_canny.timer, 1);
  cv::imshow("dist_matrix_illus", phset.get_dist_matrix_illus());
  cv::imshow("dist_matrix_colormap_caption", phset.get_dist_matrix_colormap_caption());
  cv::imshow("dist_matrix_illus_caption1", phset.get_dist_matrix_illus_caption1());
  cv::imshow("dist_matrix_illus_caption2", phset.get_dist_matrix_illus_caption2());
  cv::waitKey(0);
}

////////////////////////////////////////////////////////////////////////////////

#include "vision_utils/dgaitdb_filename.h"
void benchmark_gait_train() {
  std::string input_folder = "/home/user/Downloads/0datasets/DGaitDB_imgs/";
  DGaitDBFilename f(input_folder);
  PersonHistogramSet set;
  PersonHistogram ph;
  for (unsigned int file_idx = 1; file_idx <= DGaitDBFilename::ONI_FILES; ++file_idx) {
    printf("Loading file %i of %i\n", file_idx, DGaitDBFilename::ONI_FILES);
    for (unsigned int train_idx = 1; train_idx <= DGaitDBFilename::NFILES_TRAIN; ++train_idx) {
      // create PersonHistogram
      bool refresh_images = (train_idx == 1);
      assert(ph.create(f.filename_train(file_idx, train_idx), DGaitDBFilename::USER_IDX,
                       DEFAULT_KINECT_SERIAL(), refresh_images));
      if (refresh_images) {
        std::ostringstream fn; fn << "illus_color_img" << file_idx << ".png";
        image_utils::imwrite_debug(fn.str(), ph.get_illus_color_img());
        cv::imshow("illus_color_img", ph.get_illus_color_img());
        cv::waitKey(20);
      }
      // store it - no retrain
      set.push_back(ph, file_idx, false);
    } // end loop train_idx
  } // end loop file_idx
  assert(set.train_svm());
  assert(set.to_yaml(input_folder + "PersonHistogramSet.yaml"));
  printf("Succesfully trained...\n");
}

////////////////////////////////////////////////////////////////////////////////

void benchmark_gait_test() {
  std::string input_folder = "/home/user/Downloads/0datasets/DGaitDB_imgs/";
  DGaitDBFilename f(input_folder);
  unsigned int oni_nfiles = 55, nfiles_test_wanted = 10;
  PersonHistogramSet set;
  set.from_yaml(input_folder + "PersonHistogramSet.yaml");
  unsigned int ntests = 0, nsuccesses_svm = 0, nsuccesses_hist = 0;

  for (unsigned int file_idx = 1; file_idx <= oni_nfiles; ++file_idx) {
    printf("Training with file %i of %i\n", file_idx, oni_nfiles);
    for (unsigned int test_idx = 1; test_idx <= nfiles_test_wanted; ++test_idx) {
      // create PersonHistogram
      PersonHistogram ph;
      assert(ph.create(f.filename_test(file_idx, test_idx), DGaitDBFilename::USER_IDX,
                       DEFAULT_KINECT_SERIAL(), false));
      // test it
      int out_label_svm, out_label_hist, exp_label = file_idx;
      assert(set.compare_to(ph, out_label_hist));
      assert(set.compare_svm(ph, out_label_svm));
      ++ntests;
      if (out_label_hist == exp_label)
        ++nsuccesses_hist;
      if (out_label_svm == exp_label)
        ++nsuccesses_svm;
      //else
      printf("exp:%i, out_hist:%i, out_svm:%i\n", exp_label, out_label_hist, out_label_svm);
    } // end loop test_idx
  } // end loop file_idx
  printf("Succesfully tested, success rate: hist:%f, svm:%f\n",
         1.f * nsuccesses_hist / ntests, 1.f * nsuccesses_svm / ntests);
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
#if 0
  convert_yaml_to_images(juggling1_file);
  convert_yaml_to_images(juggling2_file);
  convert_yaml_to_images(IMG_DIR "depth/inside1");
  convert_yaml_to_images(IMG_DIR "depth/inside2");
  camera_info_bag_to_binary(KINECT_SERIAL_LAB());
  return 0;
#endif

  using namespace test_person_histogram_set_variables;
  int idx = 1;
  if (argc < 2) {
    printf("%i: test_depth_canny(IMG_DIR depth/inside1)\n", idx++);
    printf("%i: test_depth_canny(IMG_DIR depth/juggling)\n", idx++);
    printf("\n");
#if 0
    printf("%i: test_person_compute_user_mask(IMG_DIR depth/juggling1)\n", idx++);
    printf("%i: test_person_compute_user_mask(IMG_DIR depth/juggling2)\n", idx++);
    printf("%i: test_person_compute_user_mask(IMG_DIR depth/juggling3)\n", idx++);
    //  printf("\n");
    //  printf("%i :test_person_histogram(IMG_DIR depth/juggling1)\n", idx++);
    printf("\n");
    printf("%i: test_find_mask_then_top_point_centered(IMG_DIR depth/juggling1)\n", idx++);
    printf("%i: test_find_mask_then_top_point_centered(IMG_DIR depth/juggling2)\n", idx++);
    printf("%i: test_find_mask_then_top_point_centered(IMG_DIR depth/alvaro2)\n", idx++);
    printf("\n");
#endif
    printf("%i: test_person_vector_of_histograms(IMG_DIR depth/juggling1, KINECT_SERIAL_LAB(), juggling1_pt)\n", idx++);
    printf("%i: test_person_vector_of_histograms(IMG_DIR depth/juggling2, KINECT_SERIAL_LAB(), juggling2_pt)\n", idx++);
    printf("%i: test_person_vector_of_histograms(IMG_DIR depth/juggling3, KINECT_SERIAL_LAB(), juggling3_pt)\n", idx++);
    printf("%i: test_person_vector_of_histograms(IMG_DIR depth/alberto1, KINECT_SERIAL_LAB(), alberto1_pt)\n", idx++);
    printf("%i: test_person_vector_of_histograms(IMG_DIR depth/alberto2, KINECT_SERIAL_LAB(), alberto2_pt)\n", idx++);
    printf("%i: test_person_vector_of_histograms(IMG_DIR depth/alvaro1, KINECT_SERIAL_LAB(), alvaro1_pt)\n", idx++);
    printf("%i: test_person_vector_of_histograms(IMG_DIR depth/alvaro2, KINECT_SERIAL_LAB(), alvaro2_pt)\n", idx++);
    printf("\n");
    printf("%i: test_factory_from_vector(juggling_filename_prefixes, juggling_seeds, juggling_kinect_serial_numbers)\n", idx++);
    printf("%i: test_factory_from_vector(alberto_filename_prefixes, alberto_seeds, alberto_kinect_serial_numbers)\n", idx++);
    printf("%i: test_factory_from_vector(alvaro_filename_prefixes, alvaro_seeds, alvaro_kinect_serial_numbers)\n", idx++);
    printf("%i: test_factory_from_vector(all_filename_prefixes, all_seeds, all_kinect_serial_numbers)\n", idx++);
    printf("\n");
    printf("%i: test_multi_hist(all_filename_prefixes, all_seeds, all_kinect_serials)\n", idx++);
    printf("%i: test_structured_person_histogram_set();\n", idx++);
    printf("%i: test_structured_person_histogram_set_reco();\n", idx++);
    printf("\n");
    printf("%i: benchmark_gait_train()\n", idx++);
    printf("%i: benchmark_gait_test()\n", idx++);
    return -1;
  }
  int choice = 4;
  choice = atoi(argv[1]);

  idx = 1;
  if (choice == idx++)
    test_depth_canny(IMG_DIR "depth/inside1");
  else if (choice == idx++)
    test_depth_canny(juggling1_file);

#if 0
  else if (choice == idx++)
    test_person_compute_user_mask(juggling1_file, juggling1_seed);
  else if (choice == idx++)
    test_person_compute_user_mask(juggling2_file, juggling2_seed);
  else if (choice == idx++)
    test_person_compute_user_mask(juggling3_file, juggling3_seed);

  //  else if (choice == idx++)
  //    test_person_histogram(juggling1_file, juggling1_pt);

  else if (choice == idx++)
    test_find_mask_then_top_point_centered(juggling1_file, juggling1_seed);
  else if (choice == idx++)
    test_find_mask_then_top_point_centered(juggling2_file, juggling2_seed);
  else if (choice == idx++)
    test_find_mask_then_top_point_centered(alvaro2_file, alvaro2_seed);
#endif

  else if (choice == idx++)
    test_person_vector_of_histograms(juggling1_file, juggling1_seed, KINECT_SERIAL_LAB());
  else if (choice == idx++)
    test_person_vector_of_histograms(juggling2_file, juggling2_seed, KINECT_SERIAL_LAB());
  else if (choice == idx++)
    test_person_vector_of_histograms(juggling3_file, juggling3_seed, KINECT_SERIAL_LAB());
  else if (choice == idx++)
    test_person_vector_of_histograms(alberto1_file, alberto1_seed, KINECT_SERIAL_LAB());
  else if (choice == idx++)
    test_person_vector_of_histograms(alberto2_file, alberto2_seed, KINECT_SERIAL_LAB());
  else if (choice == idx++)
    test_person_vector_of_histograms(alvaro1_file, alvaro1_seed, KINECT_SERIAL_LAB());
  else if (choice == idx++)
    test_person_vector_of_histograms(alvaro2_file, alvaro2_seed, KINECT_SERIAL_LAB());

  else if (choice == idx++)
    test_factory_from_vector(juggling_filename_prefixes, juggling_seeds, juggling_kinect_serials);
  else if (choice == idx++)
    test_factory_from_vector(alberto_filename_prefixes, alberto_seeds, alberto_kinect_serials);
  else if (choice == idx++)
    test_factory_from_vector(alvaro_filename_prefixes, alvaro_seeds, alvaro_kinect_serials);
  else if (choice == idx++)
    test_factory_from_vector(refset_filename_prefixes, refset_seeds, refset_kinect_serials);

  else if (choice == idx++)
    test_multi_hist(refset_filename_prefixes, refset_seeds, refset_kinect_serials, refset_labels());
  else if (choice == idx++)
    test_structured_person_histogram_set();
  else if (choice == idx++)
    test_structured_person_histogram_set_reco();

  else if (choice == idx++)
    benchmark_gait_train();
  else if (choice == idx++)
    benchmark_gait_test();
  return 0;
}

