/*!
  \file        gtest_face_recognizer.cpp
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2013/12/10

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
#include "vision_utils/timer.h"
#include <vision_utils/img_path.h>
#include "people_recognition_vision/face_recognizer.h"
#include <people_msgs/People.h>

#define FACES_DIR   vision_utils::IMG_DIR() +  "faces/"

//#define DISPLAY

TEST(TestSuite, empty) {
  face_recognition::FaceRecognizer rec;
  EXPECT_NO_FATAL_FAILURE();
  EXPECT_TRUE(rec.nb_persons_in_model() == 0);
  EXPECT_TRUE(rec.nb_training_images() == 0);
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, people_lab_load) {
  face_recognition::FaceRecognizer rec;
  EXPECT_TRUE(rec.from_xml_file(FACES_DIR "people_lab/index.xml"));
  EXPECT_TRUE(rec.nb_persons_in_model() >= 4);
  EXPECT_TRUE(rec.nb_training_images() >= 30);
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, people_lab_from_to_xml) {
  face_recognition::FaceRecognizer rec, rec2;
  EXPECT_TRUE(rec.from_xml_file(FACES_DIR "people_lab/index.xml"));
  // save
  std::string tmp_xml = "/tmp/gtest_face_recognizer.xml";
  EXPECT_TRUE(rec.to_xml_file(tmp_xml));
  // load
  EXPECT_TRUE(rec2.from_xml_file(tmp_xml));
  EXPECT_TRUE(rec.nb_persons_in_model() == rec2.nb_persons_in_model());
  EXPECT_TRUE(rec.nb_training_images() == rec2.nb_training_images());
}


////////////////////////////////////////////////////////////////////////////////

inline void test_recognizer(face_recognition::FaceRecognizer & rec,
                            const std::string imgfile, const std::string exp_name) {
  cv::Mat3b sample = cv::imread(imgfile, CV_LOAD_IMAGE_COLOR);
  face_recognition::PersonName name = rec.predict_non_preprocessed_face(sample);
  EXPECT_TRUE(name == exp_name)
      << "img_file:'" << imgfile
      << "', name: '" << name << "', exp_name;'" << exp_name << "'";
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, people_lab_predict_known_faces) {
  face_recognition::FaceRecognizer rec;
  EXPECT_TRUE(rec.from_xml_file(FACES_DIR "people_lab/index.xml"));
  // empty image
  test_recognizer(rec, "/foo.png", "RECFAIL");
  test_recognizer(rec, FACES_DIR "people_lab/alberto_0.png", "alberto");
  test_recognizer(rec, FACES_DIR "people_lab/arnaud_20.png", "arnaud");
  test_recognizer(rec, FACES_DIR "people_lab/avi_35.png", "avi");
}

////////////////////////////////////////////////////////////////////////////////

TEST(TestSuite, google_genders_full_test) {
  face_recognition::FaceRecognizer rec;
  EXPECT_TRUE(rec.from_xml_file(FACES_DIR "google_genders/index.xml"));
  EXPECT_TRUE(rec.nb_persons_in_model() == 2); // man, woman
  // from_to_xml
  // save
  std::string tmp_xml = "/tmp/gtest_face_recognizer.xml";
  EXPECT_TRUE(rec.to_xml_file(tmp_xml));
  // load
  face_recognition::FaceRecognizer rec2;
  EXPECT_TRUE(rec2.from_xml_file(tmp_xml));
  EXPECT_TRUE(rec.nb_persons_in_model() == rec2.nb_persons_in_model());
  EXPECT_TRUE(rec.nb_training_images() == rec2.nb_training_images());
  // predict_known_faces
  for (unsigned int rec_idx = 0; rec_idx < 2; ++rec_idx) {
    face_recognition::FaceRecognizer *rec_ptr = (rec_idx ? &rec2 : &rec);
    test_recognizer(*rec_ptr, FACES_DIR "google_genders/man_0.png", "man");
    test_recognizer(*rec_ptr, FACES_DIR "google_genders/man_10.png", "man");
    test_recognizer(*rec_ptr, FACES_DIR "google_genders/man_20.png", "man");
    test_recognizer(*rec_ptr, FACES_DIR "google_genders/man_100.png", "man");
    test_recognizer(*rec_ptr, FACES_DIR "google_genders/woman_115.png", "woman");
    test_recognizer(*rec_ptr, FACES_DIR "google_genders/woman_150.png", "woman");
    test_recognizer(*rec_ptr, FACES_DIR "google_genders/woman_200.png", "woman");
    test_recognizer(*rec_ptr, FACES_DIR "google_genders/woman_260.png", "woman");
  } // end loop rec_idx
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv){
  // Run all the tests that were declared with TEST()
  ros::Time::init();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


