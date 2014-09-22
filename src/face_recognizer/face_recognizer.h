#ifndef FACE_RECOGNIZER_H
#define FACE_RECOGNIZER_H

// opencv
#include <opencv2/highgui/highgui.hpp>
// utils
#include <src/geom/rect_utils.h>
#include <src/xml/XmlDocument.h>
#include <src/map/map_utils.h>
#include <src/system/system_utils.h>
#include <src/string/filename_handling.h>
#include <src/time/timer.h>
// vision
#include <vision_utils/image_utils/opencv_face_detector.h>
#include <vision_utils/image_utils/io.h>
#include <vision_utils/image_utils/resize_utils.h>

// people_msgs
#include "people_msgs/PeoplePoseList.h"

// facerec - https://github.com/bytefish/libfacerec
// http://www.bytefish.de/blog/pca_in_opencv
//#include <vision_utils/third_parties/libfacerec/include/facerec.hpp>
#include <vision_utils/image_utils/drawing_utils.h>
#include <opencv2/contrib/contrib.hpp>
#include <people_msgs/PeoplePose.h>

namespace face_recognition {

typedef cv::Mat3b ColorImage;
typedef cv::Mat3b NonPreprocessedColorFace;
typedef cv::Mat PreprocessedBWFace; // cannot be cv::Mat1b otherwise libfacerec fails
typedef int PersonLabel;
typedef std::string PersonName;
static const PersonName NOBODY = "nobody";

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
 *  \class  FaceRecognizer
 *  A class for easily recognizing faces, based on facerec
 *  \link https://github.com/bytefish/libfacerec
 */
class FaceRecognizer {
public:

  /*! constructor */
  FaceRecognizer() {
    classifier = image_utils::create_face_classifier();
    //_model = cv::createEigenFaceRecognizer();
    _model = cv::createFisherFaceRecognizer(); // best, cf Robocity paper
    //_model = cv::createLBPHFaceRecognizer();
  }

  //////////////////////////////////////////////////////////////////////////////

  //! dtor
  ~FaceRecognizer() {
    // delete _model;
  }

  //////////////////////////////////////////////////////////////////////////////

  bool from_color_images_filenames(const std::vector<std::string> & images_filenames,
                                   const std::vector<PersonName> & names) {
    maggieDebug2("from_color_images_filenames(%i images)", images_filenames.size());
    if (images_filenames.size() != names.size()) {
      ROS_WARN("Cannot create face recognizer if nb images (%i) != nb names (%i).",
               images_filenames.size(), names.size());
      return false;
    }
    // stuff for image loading
    ColorImage color_img;
    // stuff for face detection
    cv::Mat3b small_img;
    std::vector< cv::Rect > found_faces;
    // stuff for preprocessing
    PreprocessedBWFace person_BW_face;
    std::vector<PreprocessedBWFace> faces;
    std::vector<PersonName> names_filtered;
    faces.reserve(images_filenames.size());
    names_filtered.reserve(images_filenames.size());

    // iterate on the images
    for (unsigned int img_idx = 0; img_idx < images_filenames.size(); ++img_idx) {
      if (img_idx % 100 == 0)
        maggieDebug2("from_color_images_filenames(%i images done / %i)", img_idx, images_filenames.size());
      // load_color_images
      color_img = cv::imread(images_filenames[img_idx], CV_LOAD_IMAGE_COLOR);
      if (color_img.empty()) {
        ROS_WARN("Error in reading image '%s', skipping it.",
                 images_filenames[img_idx].c_str());
        continue;
      }
      // try to find faces in images
      image_utils::detect_with_opencv(color_img, classifier, small_img, found_faces);
      if (found_faces.size() != 1) // probably some false positives -> skip
        continue;
      // pre-process results of face detection
      // to call preprocess face, we need to know face_width and face_height
      bool preprocess_success = preprocess_face
          (color_img(found_faces.front()), person_BW_face);
      if (!preprocess_success)
        continue;
      faces.push_back(person_BW_face.clone());
      names_filtered.push_back(names[img_idx]);
    } // end loop img
    // process
    return from_preprocessed_faces(faces, names_filtered);
  }

  //////////////////////////////////////////////////////////////////////////////

  bool from_color_images(const std::vector<ColorImage> & images,
                         const std::vector<PersonName> & names) {
    maggieDebug2("from_color_images(%i images)", images.size());
    if (images.size() != names.size()) {
      ROS_WARN("Cannot create face recognizer if nb images (%i) != nb names (%i).",
               images.size(), names.size());
      return false;
    }
    if(images.size() == 0) {
      ROS_WARN("Cannot create face recognizer with an empty face set.");
      return false;
    }
    std::vector<NonPreprocessedColorFace> faces_raw;
    std::vector<PersonName> names_filtered;
    names_filtered.reserve(names.size());
    faces_raw.reserve(names.size());
    // try to find faces in images
    cv::Mat3b small_img;
    std::vector< cv::Rect > found_faces;
    for (unsigned int img_idx = 0; img_idx < images.size(); ++img_idx) {
      if (img_idx % 100 == 0)
        maggieDebug2("from_color_images(%i images done / %i)", img_idx, images.size());
      image_utils::detect_with_opencv(images[img_idx], classifier, small_img,
                                      found_faces);
      // keep results of face detection
      for (unsigned int rect_idx = 0; rect_idx < found_faces.size(); ++rect_idx) {
        faces_raw.push_back(images[img_idx](found_faces[rect_idx]).clone());
        names_filtered.push_back(names[img_idx]);
      } // end loop rect
    } // end loop img

    maggieDebug2("Detected faces in %i faces out of the %i original images.",
                 faces_raw.size(), images.size());

    return from_non_preprocessed_faces(faces_raw, names_filtered);
  }

  //////////////////////////////////////////////////////////////////////////////

  bool from_non_preprocessed_faces(const std::vector<NonPreprocessedColorFace> & faces_raw,
                                   const std::vector<PersonName> & names) {
    maggieDebug2("from_non_preprocessed_faces(%i faces_raw)", faces_raw.size());
    if (faces_raw.size() != names.size()) {
      ROS_WARN("Cannot create face recognizer if nb images (%i) != nb names (%i).",
               faces_raw.size(), names.size());
      return false;
    }
    if(faces_raw.size() == 0) {
      ROS_WARN("Cannot create face recognizer with an empty face set.");
      return false;
    }

    // preprocess images
    std::vector<PreprocessedBWFace> faces;
    PreprocessedBWFace person_BW_face;
    for (unsigned int face_idx = 0; face_idx < faces_raw.size(); ++face_idx) {
      if (face_idx % 100 == 0)
        maggieDebug2("from_non_preprocessed_faces(%i images done / %i)", face_idx, faces_raw.size());
      // to call preprocess face, we need to know face_width and face_height
      bool preprocess_success = preprocess_face(faces_raw[face_idx], person_BW_face);
      if (!preprocess_success)
        continue;
      faces.push_back(person_BW_face.clone());
    } // end loop face_idx

    return from_preprocessed_faces(faces, names);
  } // end from_non_preprocessed_faces()

  //////////////////////////////////////////////////////////////////////////////

  bool from_xml_file(const std::string xml_file_absolute_path) {
    ROS_INFO("from_xml_file('%s')", xml_file_absolute_path.c_str());
    std::vector<PreprocessedBWFace> preprocessed_faces;
    std::vector<PersonName> names;
    // extract the path
    std::string faces_absolute_folder =
        StringUtils::extract_folder_from_full_path(xml_file_absolute_path);

    // load the file
    XmlDocument doc;
    bool reading_ok = doc.load_from_file(xml_file_absolute_path);
    if (!reading_ok)
      return false;

    // get person nodes
    std::vector<XmlDocument::Node*> person_nodes;
    doc.get_all_nodes_at_direction(doc.root(), "persons.person", person_nodes);
    // ROS_WARN("Got %i nodes of 'person'", person_nodes.size());

    for (PersonLabel person_idx = 0; person_idx < (int) person_nodes.size();
         ++person_idx) {
      PersonName person_name =
          doc.get_node_attribute(person_nodes[person_idx], "name", "");
      if (person_name.size() == 0) {
        ROS_WARN("Person #%i does not have her attribute 'name' set. "
                 "Skipping it.", person_idx);
        continue;
      }

      // get all images for this person
      std::vector<XmlDocument::Node*> img_nodes;
      doc.get_all_nodes_at_direction
          (person_nodes[person_idx], "img", img_nodes);
      for (unsigned int img_idx = 0; img_idx < img_nodes.size(); ++img_idx) {
        // get the relative path
        std::string img_relative_path =
            doc.get_node_attribute(img_nodes[img_idx], "path", "");
        if (img_relative_path.size() == 0) {
          ROS_WARN("Img node #%i of person '%s' does not have 'path' "
                   "attribute set. Skipping img.", img_idx, person_name.c_str());
          continue;
        }
        // build the absolute path
        std::ostringstream img_absolute_path;
        img_absolute_path << faces_absolute_folder << img_relative_path;
        //ROS_WARN("absolute_path:'%s'", absolute_path.str().c_str());
        // try to read the image
        PreprocessedBWFace curr_BW_img = cv::imread(img_absolute_path.str(),
                                                    CV_LOAD_IMAGE_GRAYSCALE);
        if (curr_BW_img.data == NULL) {
          ROS_WARN("Img '%s', referred by node #%i of person '%s' "
                   "impossible to read. Skipping img.",
                   img_absolute_path.str().c_str(), img_idx, person_name.c_str());
          continue;
        }
        //  ROS_WARN("absolute_path:'%s', curr_img:'%s'",
        //           absolute_path.str().c_str(),
        //           image_utils::infosImage(curr_BW_img).c_str());

        preprocessed_faces.push_back(curr_BW_img.clone());
        names.push_back(person_name);
      } // end loop img_idx
    } // end loop person_idx

    // try to get model
    bool was_model_loaded = false;
    XmlDocument::Node* model_node = doc.get_node_at_direction(doc.root(), "model");
    if (model_node != NULL) {
      std::string model_filename = doc.get_node_attribute(model_node, "path");
      if (model_filename.size() > 0) {
        std::string model_full_filename = faces_absolute_folder + model_filename;
        try {
          _model->load(model_full_filename);
          maggieDebug2("Model succesfully loaded from '%s'", model_full_filename.c_str());
          was_model_loaded = true;
        } catch (cv::Exception e) {
          ROS_WARN("Error while loading model '%s':'%s', will retrain model.", model_full_filename.c_str(), e.what());
        }
      }
    } // if (model_node != NULL)

    bool want_retrain_model = (!was_model_loaded);
    return from_preprocessed_faces(preprocessed_faces, names, want_retrain_model);
  } // end from_xml_file();

  //////////////////////////////////////////////////////////////////////////////

  /*!
   * \param xml_absolute_filename
   *  For instance, /foo/out.xml
   *  In the folder of the file, will also be saved the images and the model (model.yaml).
   * \return
   *  true if success
   */
  bool to_xml_file(const std::string & xml_absolute_filename) const {
    ROS_INFO("to_xml_file('%s')", xml_absolute_filename.c_str());

    std::string faces_absolute_folder =
        StringUtils::extract_folder_from_full_path(xml_absolute_filename);

    // make folder
    std::ostringstream mkdir_order;
    mkdir_order << "mkdir " << faces_absolute_folder << " --parents";
    system_utils::exec_system(mkdir_order.str());

    XmlDocument doc;
    // save model
#if 1
    std::string model_filename("model.yaml");
    _model->save(faces_absolute_folder + model_filename);
    XmlDocument::Node* model_node = doc.add_node(doc.root(), "model", "");
    doc.set_node_attribute(model_node, "path", model_filename);
#endif

    // make person nodes
    XmlDocument::Node* persons_node = doc.add_node(doc.root(), "persons", "");

    // get all existing names
    std::set<std::string> names;
    for (unsigned int label_idx = 0; label_idx < _labels.size(); ++label_idx) {
      PersonName person_name = person_label_to_person_name(_labels[label_idx]);
      if (person_name == "")
        continue;
      names.insert(person_name);
    } // end loop label

    // create corresponding nodes
    for (std::set<std::string>::iterator names_it = names.begin();
         names_it != names.end(); ++names_it) {
      // write the node
      XmlDocument::Node* person_node = doc.add_node(persons_node, "person", "");
      doc.set_node_attribute(person_node, "name", *names_it);
    }
    // doc.write_to_file("/tmp/foo.xml");

    // add all pictures to person nodes
    if(_BW_faces.size() != _labels.size()) {
      printf("%i _BW_faces != %i _labels\n", _BW_faces.size(), _labels.size());
      return false;
    }
    for (unsigned int img_BW_idx = 0; img_BW_idx < _BW_faces.size(); ++img_BW_idx) {
      PersonName person_name = person_label_to_person_name(_labels[img_BW_idx]);
      if (person_name == "")
        continue;
      // make full pathname
      std::ostringstream img_abs_path_str, img_rel_path_str;
      img_rel_path_str << person_name << "_" << img_BW_idx << ".png";
      img_abs_path_str << faces_absolute_folder << img_rel_path_str.str();
      // save image
      bool write_ok = cv::imwrite(img_abs_path_str.str(), _BW_faces[img_BW_idx]);
      if (!write_ok) {
        ROS_WARN("Impossible to save face of person '%s' into file '%s'",
                 person_name.c_str(), img_abs_path_str.str().c_str());
        return false;
      }
      // add attribute
      XmlDocument::Node* person_node =
          doc.get_node_at_direction(persons_node, "person", "name", person_name);
      if (person_node == NULL) {
        ROS_WARN("Could not find 'person' node with name '%s'",
                 person_name.c_str());
        continue;
      }
      XmlDocument::Node* img_node = doc.add_node(person_node, "img", "");
      doc.set_node_attribute(img_node, "path", img_rel_path_str.str());
    } // end loop img_BW

    doc.write_to_file(xml_absolute_filename);
    return true;
  } // end to_xml_file();

  //////////////////////////////////////////////////////////////////////////////

  PersonName predict_non_preprocessed_face
  (const NonPreprocessedColorFace & person_color_face) const {
    ROS_INFO_THROTTLE(5, "predict_non_preprocessed_face(%s)",
                      image_utils::infosImage(person_color_face).c_str());
    PreprocessedBWFace person_BW_face;
    bool preprocess_success = preprocess_face(person_color_face, person_BW_face);
    if (!preprocess_success)
      return people_msgs::PeoplePose::RECOGNITION_FAILED;
    return predict_preprocessed_face(person_BW_face);
  } // end predict_non_preprocessed_face();

  //////////////////////////////////////////////////////////////////////////////

  PersonName predict_color_image(const ColorImage & image) {
    ROS_INFO_THROTTLE(5, "predict_color_image(%s)",
                      image_utils::infosImage(image).c_str());
    // Timer timer;
    cv::Mat3b small_img;
    std::vector< cv::Rect > found_faces;
    image_utils::detect_with_opencv(image, classifier, small_img,
                                    found_faces);
    // Timer::Time t1 = timer.getTimeMilliseconds();
    // timer.reset();
    if (found_faces.size() == 0)
      return NOBODY;
    PersonName res = predict_non_preprocessed_face(image(found_faces.front()));
    //timer.printTime("predict_non_preprocessed_face()");
    //    ROS_WARN("predict_color_image(): "
    //             "width, height, img size, t_detect_faces, t_predict_non_preprocessed_face  "
    //             "%i  %i  %i  %g  %g",
    //             image.cols, image.rows, image.cols * image.rows,
    //             t1, timer.getTimeMilliseconds());
    return res;
  }

  //////////////////////////////////////////////////////////////////////////////

  void predict_color_images(const std::vector<ColorImage> & images,
                            std::vector<PersonName> & results) {
    ROS_INFO_THROTTLE(5, "predict_color_images(%i images)", images.size());
    results.resize(images.size());
    for (unsigned int img_idx = 0; img_idx < images.size(); ++img_idx) {
      if (img_idx % 100 == 0)
        maggieDebug2("predict_color_images(%i images done / %i)",
                     img_idx, images.size());
      results[img_idx] = predict_color_image(images[img_idx]);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void benchmark_color_images_filenames(const std::vector<std::string> & images_filenames,
                                        const std::vector<PersonName> & names) {
    maggieDebug2("benchmark_color_images_filenames(%i images)",
                 images_filenames.size());

    cv::Mat img;
    int success_nb = 0, failures_nb = 0, no_face_nb = 0;
    for (unsigned int img_idx = 0; img_idx < images_filenames.size(); ++img_idx) {
      if (img_idx % 100 == 0)
        maggieDebug2("predict_color_images(%i images done / %i), "
                     "success_nb:%i, failures_nb:%i, no_face_nb:%i",
                     img_idx, images_filenames.size(),
                     success_nb, failures_nb, no_face_nb);
      img = cv::imread(images_filenames[img_idx], CV_LOAD_IMAGE_COLOR);
      if (img.empty()) {
        ROS_WARN("Error in reading image '%s', skipping it.",
                 images_filenames[img_idx].c_str());
        continue;
      }
      PersonName result = predict_color_image(img);
      ROS_INFO_THROTTLE(1, "img %i ('%s') expected '%s', got '%s'",
                        img_idx, images_filenames[img_idx].c_str(),
                        names[img_idx].c_str(), result.c_str());
      if (result == NOBODY)
        ++no_face_nb;
      else if (result == names[img_idx])
        ++success_nb;
      else
        ++failures_nb;
    } // end loop img
    ROS_WARN("success_nb:%i, failures_nb:%i, no_face_nb:%i",
             success_nb, failures_nb, no_face_nb);
  } // end benchmark_color_images_filenames();

  //////////////////////////////////////////////////////////////////////////////

  /*!
   Add a new face to a given person and retrain the model
   \param person_color_face
      the image to add
   \param person_name
     the person to be updated
   \return bool
      true if succesfully added,
      false otherwise (no face found in the image, preprocessing failed...)
  */
  bool add_non_preprocessed_face_to_person
  (const NonPreprocessedColorFace & person_color_face,
   const PersonName & person_name) {
    ROS_INFO("add_non_preprocessed_face_to_person(%s: '%s')",
             image_utils::infosImage(person_color_face).c_str(),
             person_name.c_str());
    PreprocessedBWFace person_BW_face;
    bool preprocess_success = preprocess_face(person_color_face, person_BW_face);
    if (!preprocess_success)
      return false;
    return add_preprocessed_face_to_person(person_BW_face, person_name);
  } // end add_preprocessed_face_to_person();

  //////////////////////////////////////////////////////////////////////////////

  //! \return the number of persons in the model
  inline unsigned int nb_persons_in_model() const {
    return _person_labels_map.size();
  }

  //! \return the number of training images
  inline unsigned int nb_training_images() const {
    return _BW_faces.size();
  }


  //////////////////////////////////////////////////////////////////////////////

private:

  //////////////////////////////////////////////////////////////////////////////

  bool from_preprocessed_faces(const std::vector<PreprocessedBWFace> & preprocessed_faces,
                               const std::vector<PersonName> & names,
                               bool want_retrain_model = true) {
    maggieDebug2("from_preprocessed_faces(%i preprocessed faces, want_retrain_model:%i)",
                 preprocessed_faces.size(), want_retrain_model);
    _BW_faces = preprocessed_faces; // copy faces
    _person_labels_map.clear();
    _labels.clear();
    _labels.reserve(preprocessed_faces.size());
    for (unsigned int face_idx = 0; face_idx < preprocessed_faces.size(); ++face_idx)
      _labels.push_back(person_name_to_person_label(names[face_idx]));

    if (want_retrain_model)
      retrain_model();
    return true;
  } // end from_preprocessed_faces()

  //////////////////////////////////////////////////////////////////////////////

  inline bool retrain_model() {
    ROS_WARN("retrain_model(%i faces, %i labels '%s', _person_labels_map:'%s')",
             _BW_faces.size(), _labels.size(),
             StringUtils::accessible_to_string(_labels).c_str(),
             StringUtils::map_to_string(_person_labels_map).c_str());
    Timer timer;
    // check sizes
    if(_BW_faces.size() != _labels.size()) {
      printf("%i _BW_faces != %i _labels\n", _BW_faces.size(), _labels.size());
      return false;
    }
    for (unsigned int face_idx = 1; face_idx < _BW_faces.size(); ++face_idx) {
      if(_BW_faces[face_idx].size() != _BW_faces[0].size()) {
        printf("_BW_faces #%i:%ix%i != #0:%ix%i\n",
               face_idx, _BW_faces[face_idx].cols, _BW_faces[face_idx].rows,
               _BW_faces[0].cols, _BW_faces[0].rows);
        return false;
      }
    }

    //    cv::Mat out;
    //    image_utils::paste_images(_BW_faces, out, 32, 32, 0, false);
    //    cv::imshow("out", out);
    //    cv::waitKey(0);
    //  for (unsigned int face_idx = 0; face_idx < _BW_faces.size(); ++face_idx)
    //    cv::imshow(StringUtils::cast_to_string(face_idx), _BW_faces[face_idx]);
    //  cv::waitKey(0);

    try {
      _model->train(_BW_faces, _labels);
      save_back_up_file();
    } catch (cv::Exception e) {
      ROS_WARN("Exception while retrain_model():'%s'", e.what());
      return false;
    }
    ROS_INFO("Time to train model:%f ms (%i people, %i faces in total)",
             timer.getTimeMilliseconds(), nb_persons_in_model(), _BW_faces.size());
    return true;
  } // end retrain_model();

  //////////////////////////////////////////////////////////////////////////////

  inline PersonName person_label_to_person_name(const PersonLabel & person_label) const {
    PersonName person_name;
    if (map_utils::direct_search(_person_labels_map, person_label, person_name))
      return person_name;
    ROS_WARN("Label #%i does not exist.", person_label);
    return "";
  } // end person_label_to_person_name()

  //////////////////////////////////////////////////////////////////////////////

  inline PersonLabel person_name_to_person_label
  (const PersonName & person_name) {
    PersonLabel person_label;
    bool lookup_success = map_utils::reverse_search
        (_person_labels_map, person_name, person_label);
    if (lookup_success)
      return person_label;

    //    ROS_INFO("Person '%s' is not a knwon person, _person_labels_map:'%s'. Adding it.",
    //             person_name.c_str(), StringUtils::map_to_string(_person_labels_map).c_str());
    // keep the association PersonLabel <-> PersonName
    person_label = _person_labels_map.size();
    _person_labels_map.insert
        (std::pair<PersonLabel, PersonName>(person_label, person_name));
    return person_label;
  } // end person_name_to_person_label()

  //////////////////////////////////////////////////////////////////////////////

  inline void save_back_up_file() const {
    std::ostringstream xml_filename_stream;
    xml_filename_stream << "/tmp/face_recognizer_backup_" << StringUtils::timestamp()
                        << "/index.xml";
    to_xml_file(xml_filename_stream.str());
  }

  //////////////////////////////////////////////////////////////////////////////

  inline bool preprocess_face
  (const NonPreprocessedColorFace & color_face, PreprocessedBWFace & bw_face) const {
    try {
      // to B&W
      cv::cvtColor(color_face, bw_face, cv::COLOR_BGR2GRAY);
      // now scale it
      // default size 120 X 80, but use the size of the training set if different
      int face_width = (_BW_faces.size() == 0 ? 80 : _BW_faces.front().cols);
      int face_height = (_BW_faces.size() == 0 ? 120 : _BW_faces.front().rows);
      image_utils::resize_constrain_proportions(bw_face, bw_face, face_width, face_height);
      // equalize histogram
      cv::equalizeHist(bw_face, bw_face);
      // show image
      //cv::imshow("color_face", color_face); cv::imshow("bw_face", bw_face); cv::waitKey(1000);
    } catch (cv::Exception e) {
      ROS_WARN("Error while preprocessing face:'%s'", e.what());
      return false;
    }
    return true;
  } // end preprocess_face()

  //////////////////////////////////////////////////////////////////////////////

  PersonName predict_preprocessed_face(const PreprocessedBWFace & face) const {
    ROS_INFO_THROTTLE(5, "predict_preprocessed_face(%s)",
                      image_utils::infosImage(face).c_str());

    //PersonLabel person_label = _model->predict(face);
    double confidence;
    PersonLabel person_label;
    try {
      _model->predict(face, person_label, confidence);
    } catch (cv::Exception e) {
      ROS_WARN("Error while predictiong with model:'%s'", e.what());
      return people_msgs::PeoplePose::RECOGNITION_FAILED;
    }
    //maggieDebug2("predict: person_label:%i, confidence:%g", person_label, confidence);

    PersonName person_name;
    bool lookup_success = map_utils::direct_search
        (_person_labels_map, person_label, person_name);
    if (!lookup_success) {
      ROS_WARN("The prediction is label #%i but it does not correspond to a "
               "knwon person, _person_labels_map:'%s'",
               person_label, StringUtils::map_to_string(_person_labels_map).c_str());
      return people_msgs::PeoplePose::RECOGNITION_FAILED;
    }
    return person_name;
  }

  //////////////////////////////////////////////////////////////////////////////

  /*!
   \param person_BW_face
   \param person_name
   \return bool
  */
  bool add_preprocessed_face_to_person
  (const PreprocessedBWFace & person_BW_face,
   const PersonName & person_name)
  {
    ROS_INFO("add_preprocessed_face_to_person(%s: '%s')",
             image_utils::infosImage(person_BW_face).c_str(),
             person_name.c_str());

    // get the person label
    PersonLabel person_label = person_name_to_person_label(person_name);
    // add it to our data base
    _BW_faces.push_back(person_BW_face.clone());
    _labels.push_back(person_label);
    // re-train the model
    ROS_INFO("Retraining the new model with this face.");
    retrain_model();
    return true;
  } // end add_preprocessed_face_to_person();

  //////////////////////////////////////////////////////////////////////////////

  //! the classifier to find the faces
  cv::CascadeClassifier classifier;
  //! the faces in black and white
  std::vector<PreprocessedBWFace> _BW_faces;
  //int face_height, face_width;
  //! for each face, the label of the person it corresponds to
  std::vector<PersonLabel> _labels;
  //! the associations PersonLabel <-> PersonName
  std::map<PersonLabel, PersonName> _person_labels_map;
  //! the trained model
  cv::Ptr<cv::FaceRecognizer> _model;
}; // end class FaceRecognizer

} // end namespace face_recognition

#endif // FACE_RECOGNIZER_H
