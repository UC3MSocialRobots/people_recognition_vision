/*!
  \file        breast_detector.cpp
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2013/10/15

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

\section Parameters
  - \b "foo"
        [string] (default: "bar")
        Description of the parameter.

\section Subscriptions
  - \b "/foo"
        [xxx]
        Descrption of the subscription

\section Publications
  - \b "~foo"
        [xxx]
        Descrption of the publication

 */

#include "vision_utils/nite_subscriber_template.h"
#include "people_recognition_vision/breast_detector.h"
#include "vision_utils/dgaitdb_filename.h"

class NiteBreastDetector : public NiteSubscriberTemplate {
public:

  virtual void init() {
    NiteSubscriberTemplate::init();
    want_illus = true;
  }

  void fn(const cv::Mat3b & color,
          const cv::Mat1f & depth,
          const cv::Mat1b & user,
          const kinect::NiteSkeletonList & skeleton_list) {
    ROS_INFO_THROTTLE(1, "NiteBreastDetector:fn()");
    bool ok = detector.breast_all_values
              (depth, user, depth_camera_model, heights,
               BreastDetector::WALK3D, want_illus, &breasts_illus);
    if (!ok)
      return;
    cv::imshow("breasts_illus", breasts_illus);
    char c= cv::waitKey(10);
    if (c == 'i')
      want_illus = !want_illus;
  } // end image_callback();

private:
  std::map<int, BreastDetector::HeightBreast> heights;
  bool want_illus;
  BreastDetector detector;
  cv::Mat3b breasts_illus;
}; // end class NiteBreastDetector

void test_nite() {
  NiteBreastDetector skill;
  skill.init();
  ros::spin();
}

////////////////////////////////////////////////////////////////////////////////

void train_DGaitDB(BreastDetector::Method method = BreastDetector::WALK3D) {
  DGaitDBFilename f("/home/user/Downloads/0datasets/DGaitDB_imgs/");
  // load data
  unsigned int nfiles = DGaitDBFilename::ONI_FILES * DGaitDBFilename::NFILES_TRAIN;
  image_geometry::PinholeCameraModel depth_camera_model, rgb_camera_model;
  kinect_openni_utils::read_camera_model_files(DEFAULT_KINECT_SERIAL(), depth_camera_model, rgb_camera_model);
  // now train
  BreastDetector detector;
  
  const uchar USER_IDX_auxConst = DGaitDBFilename::USER_IDX;
  
  bool ok = detector.train
            (f.all_filenames_train(),
             std::vector<uchar> (nfiles, USER_IDX_auxConst),
             std::vector<image_geometry::PinholeCameraModel> (nfiles, depth_camera_model),
             f.all_genders_train<BreastDetector::Gender>(BreastDetector::MALE, BreastDetector::FEMALE),
             method);
  if (!ok)
    printf("BreastDetector failed to train.\n");
  else
    printf("BreastDetector succesfully trained.\n");
}

////////////////////////////////////////////////////////////////////////////////

bool test_DGaitDB(BreastDetector::Method method = BreastDetector::WALK3D) {
  DGaitDBFilename f("/home/user/Downloads/0datasets/DGaitDB_imgs/");
  // load data
  unsigned int nfiles = DGaitDBFilename::ONI_FILES * DGaitDBFilename::NFILES_TEST;
  image_geometry::PinholeCameraModel depth_camera_model, rgb_camera_model;
  kinect_openni_utils::read_camera_model_files(DEFAULT_KINECT_SERIAL(), depth_camera_model, rgb_camera_model);
  std::vector<std::string> files = f.all_filenames_test();
  std::vector<BreastDetector::Gender> exp_genders =
      f.all_genders_test<BreastDetector::Gender>(BreastDetector::MALE, BreastDetector::FEMALE);

  BreastDetector detector;
  int nsuccesses = 0, nfiles_tested = 0;
  for (unsigned int file_idx = 0; file_idx < nfiles; ++file_idx) {
    if (file_idx % 100 == 0)
      printf(" --> tested %i files of %i, %i successes = %g %%\n",
             file_idx, nfiles, nsuccesses, 1. * nsuccesses / file_idx);
    cv::Mat1b user_mask;
    cv::Mat3b rgb;
    cv::Mat1f depth;
    if (!image_utils::read_rgb_depth_user_image_from_image_file
        (files[file_idx], &rgb, &depth, &user_mask))
      continue;
      
    const uchar USER_IDX_auxConst = DGaitDBFilename::USER_IDX;  
      
    user_mask = (user_mask == USER_IDX_auxConst);
    Timer timer;
    BreastDetector::HeightBreast h = detector.detect_breast(depth, user_mask, depth_camera_model, method);
    if (h.gender == BreastDetector::ERROR) {
      printf("BreastDetector failed to test with img '%s'.\n", files[file_idx].c_str());
      continue;
    }
    // illus
    cv::Mat3b breast_illus;
    detector.breast2img(user_mask, breast_illus, BreastDetector::HeightBreast(), method);
    cv::imshow("rgb", rgb);
    cv::imshow("depth", image_utils::depth2viz(depth));
    //cv::imshow("depth", image_utils::in(depth));
    cv::imshow("breast_illus", breast_illus); cv::waitKey(5);

    BreastDetector::Gender exp_gender = exp_genders[file_idx];
    printf("Time: %g ms, computed gender %i, expected gender %i\n",
           timer.getTimeMilliseconds(), h.gender, exp_gender);
    ++nfiles_tested;
    nsuccesses += (exp_gender == h.gender ? 1 : 0);
  } // end loop file_idx
  double success_rate = 1. * nsuccesses / nfiles_tested;
  printf("BreastDetector succesfully tested, success_rate:%g.\n", success_rate);
  return true;
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  ros::init(argc, argv, "BreastDetector");
  int idx = 1;
  if (argc < 2) {
    printf("%i: test_nite()\n", idx++);
    //printf("%i: test_svm()\n", idx++);
    printf("%i: train_DGaitDB(WALK3D)\n", idx++);
    printf("%i: train_DGaitDB(REPROJECT)\n", idx++);
    printf("%i: train_DGaitDB(TEMPLATE_MATCHING)\n", idx++);
    printf("%i: test_DGaitDB(WALK3D)\n", idx++);
    printf("%i: test_DGaitDB(REPROJECT)\n", idx++);
    printf("%i: test_DGaitDB(TEMPLATE_MATCHING)\n", idx++);
    return -1;
  }

  int choice = 0;
  choice = atoi(argv[1]);

  idx = 1;
  if (choice == idx++)
    test_nite();
  //else if (choice == idx++)
  //  test_svm();
  else if (choice == idx++)
    train_DGaitDB(BreastDetector::WALK3D);
  else if (choice == idx++)
    train_DGaitDB(BreastDetector::REPROJECT);
  else if (choice == idx++)
    train_DGaitDB(BreastDetector::TEMPLATE_MATCHING);
  else if (choice == idx++)
    test_DGaitDB(BreastDetector::WALK3D);
  else if (choice == idx++)
    test_DGaitDB(BreastDetector::REPROJECT);
  else if (choice == idx++)
    test_DGaitDB(BreastDetector::TEMPLATE_MATCHING);
  return 0;
}
