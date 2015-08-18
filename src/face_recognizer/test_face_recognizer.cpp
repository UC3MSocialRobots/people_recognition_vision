#include "people_recognition_vision/face_recognizer.h"

#define FACES_DIR   IMG_DIR "faces/"


/*!
 * Split a vector into two subvectors,
 * with the elements being randomly associated to one or the other.
 * \param in
 * \param out1, out2
 *    The two output vectors.
 *    Each element of in will be in exactly one of them, randomly assignated.
 * \param ratio_in2out1
 *    The approximate ratio of in to be put in out1 (between 0 and 1)
 *    If it equals 0, in -> out2.
 *    If it equals 1, in -> out1.
 */
template<class _T>
inline void split_vector_in_two(const std::vector<_T> & in,
                                std::vector<_T> & out1,
                                std::vector<_T> & out2,
                                const double ratio_in2out1 = .5) {
  out1.clear();
  out1.reserve(ratio_in2out1 * in.size());
  out2.clear();
  out2.reserve((1.f - ratio_in2out1) * in.size());
  for (unsigned int in_idx = 0; in_idx < in.size(); ++in_idx) {
    if (drand48() <= ratio_in2out1)
      out1.push_back(in[in_idx]);
    else
      out2.push_back(in[in_idx]);
  } // end loop in
} // end split_vector_in_two()

////////////////////////////////////////////////////////////////////////////////

/*!
 * Remove elements of a vector randomly so as to keep more or less
 * a given ratio of the input elements.
 * \param in
 *    The vector to be decimated
 * \param out
 *    The smaller vector
 * \param keep_ratio
 *    If = 1, keep everything.
 *    If = 0, remove everything.
 *    If = .5, keep roughly half of the elements.
 */
template<class _T>
inline void decimate_vector(const std::vector<_T> & in,
                            std::vector<_T> & out,
                            const double keep_ratio = 1.f / 2) {
  out.clear();
  out.reserve(in.size() * keep_ratio);
  for (unsigned int in_idx = 0; in_idx < in.size(); ++in_idx) {
    if (drand48() < keep_ratio) // keep it
      out.push_back(in[in_idx]);
  } // end loop in
} // end decimate_vector()

////////////////////////////////////////////////////////////////////////////////

inline std::vector<std::string> create_google_genders_images_filenames() {
  std::vector<std::string> images_filenames;
  for (unsigned int man_idx = 0; man_idx <= 180; ++man_idx) {
    std::ostringstream filename;
    filename << "/home/user/Downloads/google_genders/man/images(" << man_idx << ")";
    images_filenames.push_back(filename.str());
    // names.push_back("man");
  } // end loop man
  for (unsigned int woman_idx = 0; woman_idx <= 379; ++woman_idx) {
    std::ostringstream filename;
    filename << "/home/user/Downloads/google_genders/woman/images(" << woman_idx << ")";
    images_filenames.push_back(filename.str());
    // names.push_back("woman");
  } // end loop woman

  return images_filenames;
}

////////////////////////////////////////////////////////////////////////////////

inline std::vector<std::string> create_yale_images_filenames() {
  std::vector<std::string> images_filenames;
  StringUtils::retrieve_file_split("/home/user/Downloads/ExtendedYaleB/index.txt",
                                   images_filenames);
  //ROS_WARN("n_pics:%i", images_filenames.size());
  return images_filenames;
}

////////////////////////////////////////////////////////////////////////////////

inline int n_men(const std::vector<face_recognition::PersonName> & names) {
  int ans = 0;
  for (unsigned int name_idx = 0; name_idx < names.size(); ++name_idx) {
    if (names[name_idx] == "man")
      ++ans;
  } // end loop name
  return ans;
}

////////////////////////////////////////////////////////////////////////////////

inline std::vector<face_recognition::PersonName> image_filenames_to_names
(const std::vector<std::string> & images_filenames,
 bool (*is_woman_ptr)(const std::string &))
{
  std::vector<face_recognition::PersonName> names;
  names.reserve(images_filenames.size());
  for (unsigned int file_idx = 0; file_idx < images_filenames.size(); ++file_idx) {
    bool is_woman = is_woman_ptr(images_filenames[file_idx]);
    names.push_back(is_woman ? "woman" : "man");
  } // end loop filename
  maggiePrint("image_filenames_to_names(): %i pics, %i%% men",
              names.size(), n_men(names) * 100 / names.size());
  return names;
}

////////////////////////////////////////////////////////////////////////////////

inline bool google_genders_is_woman(const std::string & filename) {
  return (filename.find("woman") != std::string::npos);
}

////////////////////////////////////////////////////////////////////////////////

inline bool yale_is_woman(const std::string & filename) {
  return (filename.find("yaleB15/") != std::string::npos)
      || (filename.find("yaleB22/") != std::string::npos)
      || (filename.find("yaleB27/") != std::string::npos)
      || (filename.find("yaleB28/") != std::string::npos)
      || (filename.find("yaleB34/") != std::string::npos)
      || (filename.find("yaleB37/") != std::string::npos);
}

////////////////////////////////////////////////////////////////////////////////

void train_google_genders_face_recognizer() {
  maggieDebug2("train_google_genders_face_recognizer()");
  face_recognition::FaceRecognizer reco;
  reco.from_color_images_filenames
      (create_google_genders_images_filenames(),
       image_filenames_to_names(create_google_genders_images_filenames(),
                                &google_genders_is_woman));
  reco.to_xml_file(FACES_DIR "google_genders/index.xml");
}

////////////////////////////////////////////////////////////////////////////////

void train_yale_face_recognizer() {
  maggieDebug2("train_yale_face_recognizer()");
  std::vector<std::string> filenames_big = create_yale_images_filenames();
  std::vector<std::string> filenames_small;
  decimate_vector(filenames_big, filenames_small, 1.f / 20);
  face_recognition::FaceRecognizer reco;
  reco.from_color_images_filenames
      (filenames_small, image_filenames_to_names(filenames_small, &yale_is_woman));
  reco.to_xml_file(FACES_DIR "YaleB/index.xml");
}

////////////////////////////////////////////////////////////////////////////////

void train_yale_small_face_recognizer() {
  maggieDebug2("train_yale_small_face_recognizer()");
#if 0 // create training and test subsets
  std::vector<std::string> filenames_big = create_yale_images_filenames(),
      filenames_small, training_set_filenames, test_set_filenames;
  decimate_vector(filenames_big, filenames_small, 1.f / 10);
  ROS_WARN("filenames size:%i -> %i", filenames_big.size(), filenames_small.size());

  // make both sets
  split_vector_in_two(filenames_small, training_set_filenames, test_set_filenames, 1 / 2.f);
  ROS_WARN("training_set size:%i, test_set size:%i",
           training_set_filenames.size(), test_set_filenames.size());
  StringUtils::save_file_split("/home/user/Downloads/ExtendedYaleB/training.txt",
                               training_set_filenames);
  StringUtils::save_file_split("/home/user/Downloads/ExtendedYaleB/test.txt",
                               test_set_filenames);
#else
  std::vector<std::string> training_set_filenames;
  StringUtils::retrieve_file_split("/home/user/Downloads/ExtendedYaleB/training.txt",
                                   training_set_filenames);
#endif

  // now train!
  std::vector<face_recognition::PersonName> training_set_names =
      image_filenames_to_names(training_set_filenames, &yale_is_woman);
  face_recognition::FaceRecognizer reco;
  reco.from_color_images_filenames
      (training_set_filenames, training_set_names);
  reco.to_xml_file(FACES_DIR "Yale_B_small/index.xml");
}

////////////////////////////////////////////////////////////////////////////////

void test_people_lab_face_recognizer() {
  maggieDebug2("test_people_lab_face_recognizer()");
  face_recognition::FaceRecognizer reco;
  reco.from_xml_file(FACES_DIR "people_lab/index.xml");

  // test it
  cv::Mat3b test = cv::imread(
        //IMG_DIR "arnaud001.png",
        FACES_DIR "people_lab/avi_35.png",
        CV_LOAD_IMAGE_COLOR);
  face_recognition::PersonName predict_result =
      reco.predict_non_preprocessed_face(test);
  ROS_WARN("predict_result:'%s'", predict_result.c_str());

  // add a new face
  cv::Mat3b new_pic = cv::imread(IMG_DIR "arnaud001.png", CV_LOAD_IMAGE_COLOR);
  reco.add_non_preprocessed_face_to_person(new_pic, "arnaud");

  // save file
  reco.to_xml_file(FACES_DIR "people_lab_out/index.xml");
}

////////////////////////////////////////////////////////////////////////////////

void test_google_genders_face_recognizer() {
  maggieDebug2("test_google_genders_face_recognizer()");
  face_recognition::FaceRecognizer reco;
  reco.from_xml_file(FACES_DIR "google_genders/index.xml");

  // test it
  cv::Mat3b test = cv::imread(IMG_DIR "arnaud001.png", CV_LOAD_IMAGE_COLOR);
  face_recognition::PersonName predict_result =
      reco.predict_color_image(test);
  ROS_WARN("predict_result:'%s'", predict_result.c_str());
}

////////////////////////////////////////////////////////////////////////////////

void test_google_genders_face_recognizer_varying_size() {
  maggieDebug2("test_google_genders_face_recognizer_varying_size()");
  face_recognition::FaceRecognizer reco;
  reco.from_xml_file(FACES_DIR "google_genders/index.xml");

  // test it
  cv::Mat3b test = cv::imread
      ("/home/user/Downloads/ExtendedYaleB/yaleB11/yaleB11_P00A-005E-10.pgm",
       CV_LOAD_IMAGE_COLOR);
  cv::Mat3b test_small;
  for (double ratio = 0.1; ratio < 2.; ratio+=.1) {
    cv::resize(test, test_small, cv::Size(), ratio, ratio, CV_INTER_LANCZOS4);
    face_recognition::PersonName predict_result =
        reco.predict_color_image(test_small);
    ROS_WARN("predict_result:'%s'", predict_result.c_str());
  } // end loop ratio
}

////////////////////////////////////////////////////////////////////////////////

//! benchmark: train on google images, test on yale
void benchmark_google_with_yale() {
  maggieDebug2("benchmark_google_with_yale()");
  face_recognition::FaceRecognizer reco;
  reco.from_xml_file(FACES_DIR "google_genders/index.xml");
  reco.benchmark_color_images_filenames
      (create_yale_images_filenames(),
       image_filenames_to_names(create_yale_images_filenames(), &yale_is_woman));
}

////////////////////////////////////////////////////////////////////////////////

//! benchmark: train on people from the lab, test on yale
void benchmark_yale_with_google() {
  maggieDebug2("benchmark_yale_with_google()");
  face_recognition::FaceRecognizer reco;
  reco.from_xml_file(FACES_DIR "YaleB/index.xml");
  reco.benchmark_color_images_filenames
      (create_google_genders_images_filenames(),
       image_filenames_to_names(create_google_genders_images_filenames(),
                                &google_genders_is_woman));
}

////////////////////////////////////////////////////////////////////////////////

//! benchmark: train on people from the lab, test on yale
void benchmark_yale_small_with_yale_small2() {
  maggieDebug2("benchmark_yale_small_with_yale_small2()");
  face_recognition::FaceRecognizer reco;
  reco.from_xml_file(FACES_DIR "Yale_B_small/index.xml");

  std::vector<std::string> test_filenames;
  StringUtils::retrieve_file_split("/home/user/Downloads/ExtendedYaleB/test.txt",
                                   test_filenames);
  reco.benchmark_color_images_filenames
      (test_filenames, image_filenames_to_names(test_filenames, &yale_is_woman));
}

////////////////////////////////////////////////////////////////////////////////

int main(/*int argc, char** argv*/) {
  maggieDebug2("test_face_recognizer()");
  ros::Time::init();

  int idx = 1;
  printf("%i: train_google_genders_face_recognizer();\n", idx++);
  printf("%i: train_yale_face_recognizer();\n", idx++);
  printf("%i: train_yale_small_face_recognizer();\n", idx++);
  printf("\n");

  printf("%i: test_people_lab_face_recognizer();\n", idx++);
  printf("%i: test_google_genders_face_recognizer();\n", idx++);
  printf("%i: test_google_genders_face_recognizer_varying_size();\n", idx++);
  printf("\n");

  printf("%i: benchmark_google_with_yale();\n", idx++);
  printf("%i: benchmark_yale_with_google();\n", idx++);
  printf("%i: benchmark_yale_small_with_yale_small2();\n", idx++);
  printf("\n");

  printf("Choice?\n");
  int choice;
  std::cin >> choice;

  idx = 1;
  if (choice == idx++)
    train_google_genders_face_recognizer();
  else if (choice == idx++)
    train_yale_face_recognizer();
  else if (choice == idx++)
    train_yale_small_face_recognizer();

  else if (choice == idx++)
    test_people_lab_face_recognizer();
  else if (choice == idx++)
    test_google_genders_face_recognizer();
  else if (choice == idx++)
    test_google_genders_face_recognizer_varying_size();

  else if (choice == idx++)
    benchmark_google_with_yale();
  else if (choice == idx++)
    benchmark_yale_with_google();
  else if (choice == idx++)
    benchmark_yale_small_with_yale_small2();

  return 0;
}

