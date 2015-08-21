/*!
  \file        height_benchmark.cpp
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2014/11/17

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

A benchmark for our homebrew HeightDetector vs other
state-of-the-art height detectors.
 */
#include "people_recognition_vision/height_detector.h"
#include "vision_utils/utils/Rect3.h"
typedef double Height;
typedef cv::Point3f Pt3f;
typedef geometry_utils::Rect3f Rect3f;
HeightDetector _detector;
std::vector<Pt3f> _user3D;

////////////////////////////////////////////////////////////////////////////////

Height naive_height_meters(const cv::Mat1f & depth,
                           const cv::Mat1b & user_mask,
                           const image_geometry::PinholeCameraModel & depth_camera_model) {
  // reproject mask
  bool ok = kinect_openni_utils::pixel2world_depth
            (depth, depth_camera_model, _user3D, 1, user_mask);
  if (!ok)
    return HeightDetector::ERROR;
  Rect3f bbox3D = geometry_utils::boundingBox_vec3<float, Pt3f, std::vector<Pt3f> >(_user3D);
  // printf("bbox3D:'%s'\n", bbox3D.to_string().c_str());
  return bbox3D.height;
}

////////////////////////////////////////////////////////////////////////////////

bool benchmark(const cv::Mat1f & depth,
               const cv::Mat1b & user_mask,
               const image_geometry::PinholeCameraModel & depth_camera_model,
               const Height ground_truth_height_m,
               double & my_error, double & naive_height_error) {
  HeightDetector::Height h =
      _detector.height_meters(depth, user_mask, depth_camera_model);
  if (h.height_m == HeightDetector::ERROR) {
    printf("Error using HeightDetector\n");
    return false;
  }
  my_error = fabs(h.height_m - ground_truth_height_m);
  // naive
  Height naive_height_m = naive_height_meters
                          (depth, user_mask, depth_camera_model);
  if (naive_height_m == HeightDetector::ERROR) {
    printf("Error using naive_height_meters()\n");
    return false;
  }
  naive_height_error = fabs(naive_height_m - ground_truth_height_m);
  printf("benchmark(): ground truth: %i cm, my method:%i cm (error:%i cm), "
         "naive:%i cm  (error:%i cm)\n",
         (int) (100*ground_truth_height_m),
         (int) (100*h.height_m), (int) (100*my_error),
         (int) (100*naive_height_m), (int) (100*naive_height_error));
  return true;
}

////////////////////////////////////////////////////////////////////////////////

void print_help_and_exit(int argc, char**argv) {
  printf("Synopsis: %s [] FILE where FILE is a text file "
         "where each line has the following structure:\n",
         argv[0]);
  printf("FILENAME_PREFIX  HEIGHT_METERS [USER_VALUE] [KINECT_SERIAL]\n");
  printf("FILENAME_PREFIX  HEIGHT_METERS [USER_VALUE] [KINECT_SERIAL]\n");
  printf("...\n");
  exit(-1);
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char**argv) {
  cv::Mat depth;
  cv::Mat1b user_mask;
  if (argc != 2)
    print_help_and_exit(argc, argv);
  std::string input_file(argv[1]),
      folder = StringUtils::extract_folder_from_full_path(input_file);
  std::vector<std::string> lines;
  StringUtils::retrieve_file_split(input_file, lines, true);
  unsigned int nlines = lines.size();
  if (nlines == (unsigned int) 0)
    print_help_and_exit(argc, argv);

  double sum_my_error = 0, sum_naive_height_error = 0, my_error, naive_height_error;
  int nsuccess = 0;
  for (unsigned int i = 1; i < nlines; ++i) {
    // parse line
    std::string line(lines[i]);
    if (line.size() >= 2 && line.substr(0, 2) == "//") // comment line
      continue;
    std::vector<std::string> words;
    StringUtils::StringSplit(line, " ", &words);
    if (words.size() < 2 || words.size() > 4) {
      printf("Line #%i '%s' does not respect the syntax! Skipping.\n",
             i, line.c_str());
      continue;
    }
    // read camera model
    std::string kinect_serial_number = KINECT_SERIAL_LAB();
    if (words.size() == 4) {
      std::string kinect_serial_number = words.back();
      if (kinect_serial_number == "KINECT_SERIAL_LAB")
        kinect_serial_number = KINECT_SERIAL_LAB();
      else if (kinect_serial_number == "KINECT_SERIAL_ARNAUD")
        kinect_serial_number = KINECT_SERIAL_ARNAUD();
    }
    image_geometry::PinholeCameraModel rgb_camera_model, depth_camera_model;
    assert(kinect_openni_utils::read_camera_model_files
           (kinect_serial_number, depth_camera_model, rgb_camera_model));

    // read input depth and user images
    std::string filename_prefix = folder + std::string(words[0]);
    printf("Reading file '%s'\n", filename_prefix.c_str());
    if (!image_utils::read_rgb_depth_user_image_from_image_file
        (filename_prefix, NULL, &depth, &user_mask)) {
      printf("Could not read file '%s'\n", filename_prefix.c_str());
      continue;
    }
    Height ground_truth_height_m = StringUtils::cast_from_string<Height>(words[1]);
    if (words.size() == 3) {
      unsigned char user_value = StringUtils::cast_from_string<int>(words[2]);
      // printf("Applying user value %i\n", (int) user_value);
      user_mask = (user_mask == user_value);
      //cv::imshow("user_mask", user_mask); cv::waitKey(0);
    }
    if (!benchmark(depth, user_mask, depth_camera_model, ground_truth_height_m,
                   my_error, naive_height_error))
      continue;
    ++nsuccess;
    sum_my_error += my_error;
    sum_naive_height_error += naive_height_error;
  }
  printf("Average error on %i cases:HeightDetector: %g m, naive:%g m\n",
         nsuccess, sum_my_error / nsuccess, sum_naive_height_error / nsuccess);
  return 0;
} // end main
