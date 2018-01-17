#ifndef FACE_RECOGNIZER_H
#define FACE_RECOGNIZER_H

// opencv
#include <set>
#include <opencv2/highgui/highgui.hpp>
// facerec - https://github.com/bytefish/libfacerec
// http://www.bytefish.de/blog/pca_in_opencv
#if CV_MAJOR_VERSION == 3
#include <opencv2/face/facerec.hpp>
namespace cv_face = cv::face;
#else // OpenCV 2.*
#include <opencv2/contrib/contrib.hpp>
namespace cv_face = cv;
#endif
// utils
#include "vision_utils/accessible_to_string.h"
#include "vision_utils/exec_system.h"
#include "vision_utils/extract_folder_from_full_path.h"
#include "vision_utils/infosimage.h"
#include "vision_utils/opencv_face_detector.h"
#include "vision_utils/map_to_string.h"
#include "vision_utils/map_direct_search.h"
#include "vision_utils/map_reverse_search.h"
#include "vision_utils/resize_constrain_proportions.h"
#include "vision_utils/timer.h"
#include "vision_utils/timestamp.h"
#include "vision_utils/XmlDocument.h"

namespace face_recognition {

typedef cv::Mat3b ColorImage;
typedef cv::Mat3b NonPreprocessedColorFace;
typedef cv::Mat PreprocessedBWFace; // cannot be cv::Mat1b otherwise libfacerec fails
typedef int PersonLabel;
typedef std::string PersonName;
static const PersonName NOBODY = "nobody";
static const PersonName RECOGNITION_FAILED = "RECFAIL";

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
    classifier = vision_utils::create_face_classifier();
    //_model = cv_face::createEigenFaceRecognizer();
#if CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION >= 3 // from 3.3.1 onwards
    _model = cv_face::FisherFaceRecognizer::create();
#else
    _model = cv_face::createFisherFaceRecognizer(); // best, cf Robocity paper
#endif
    //_model = cv_face::createLBPHFaceRecognizer();
  }

  //////////////////////////////////////////////////////////////////////////////

  //! dtor
  ~FaceRecognizer() {
    // delete _model;
  }

  //////////////////////////////////////////////////////////////////////////////

  bool from_color_images_filenames(const std::vector<std::string> & images_filenames,
                                   const std::vector<PersonName> & names) {
    printf("from_color_images_filenames(%li images)", images_filenames.size());
    if (images_filenames.size() != names.size()) {
      printf("Cannot create face recognizer if nb images (%li) != nb names (%li).\n",
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
        printf("from_color_images_filenames(%i images done / %li)", img_idx, images_filenames.size());
      // load_color_images
      color_img = cv::imread(images_filenames[img_idx], CV_LOAD_IMAGE_COLOR);
      if (color_img.empty()) {
        printf("Error in reading image '%s', skipping it.\n",
               images_filenames[img_idx].c_str());
        continue;
      }
      // try to find faces in images
      vision_utils::detect_with_opencv(color_img, classifier, small_img, found_faces);
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
    printf("from_color_images(%li images)", images.size());
    if (images.size() != names.size()) {
      printf("Cannot create face recognizer if nb images (%li) != nb names (%li).\n",
             images.size(), names.size());
      return false;
    }
    if(images.size() == 0) {
      printf("Cannot create face recognizer with an empty face set.\n");
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
        printf("from_color_images(%i images done / %li)", img_idx, images.size());
      vision_utils::detect_with_opencv(images[img_idx], classifier, small_img,
                                       found_faces);
      // keep results of face detection
      for (unsigned int rect_idx = 0; rect_idx < found_faces.size(); ++rect_idx) {
        faces_raw.push_back(images[img_idx](found_faces[rect_idx]).clone());
        names_filtered.push_back(names[img_idx]);
      } // end loop rect
    } // end loop img

    printf("Detected faces in %li faces out of the %li original images.",
           faces_raw.size(), images.size());

    return from_non_preprocessed_faces(faces_raw, names_filtered);
  }

  //////////////////////////////////////////////////////////////////////////////

  bool from_non_preprocessed_faces(const std::vector<NonPreprocessedColorFace> & faces_raw,
                                   const std::vector<PersonName> & names) {
    printf("from_non_preprocessed_faces(%li faces_raw)", faces_raw.size());
    if (faces_raw.size() != names.size()) {
      printf("Cannot create face recognizer if nb images (%li) != nb names (%li).\n",
             faces_raw.size(), names.size());
      return false;
    }
    if(faces_raw.size() == 0) {
      printf("Cannot create face recognizer with an empty face set.\n");
      return false;
    }

    // preprocess images
    std::vector<PreprocessedBWFace> faces;
    PreprocessedBWFace person_BW_face;
    for (unsigned int face_idx = 0; face_idx < faces_raw.size(); ++face_idx) {
      if (face_idx % 100 == 0)
        printf("from_non_preprocessed_faces(%i images done / %li)", face_idx, faces_raw.size());
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
    printf("from_xml_file('%s')\n", xml_file_absolute_path.c_str());
    std::vector<PreprocessedBWFace> preprocessed_faces;
    std::vector<PersonName> names;
    // extract the path
    std::string faces_absolute_folder =
        vision_utils::extract_folder_from_full_path(xml_file_absolute_path);

    // load the file
    vision_utils::XmlDocument doc;
    bool reading_ok = doc.load_from_file(xml_file_absolute_path);
    if (!reading_ok)
      return false;

    // get person nodes
    std::vector<vision_utils::XmlDocument::Node*> person_nodes;
    doc.get_all_nodes_at_direction(doc.root(), "persons.person", person_nodes);
    // printf("Got %li nodes of 'person'\n", person_nodes.size());

    for (PersonLabel person_idx = 0; person_idx < (int) person_nodes.size();
         ++person_idx) {
      PersonName name =
          doc.get_node_attribute(person_nodes[person_idx], "name", "");
      if (name.size() == 0) {
        printf("Person #%i does not have her attribute 'name' set. "
               "Skipping it.\n", person_idx);
        continue;
      }

      // get all images for this person
      std::vector<vision_utils::XmlDocument::Node*> img_nodes;
      doc.get_all_nodes_at_direction
          (person_nodes[person_idx], "img", img_nodes);
      for (unsigned int img_idx = 0; img_idx < img_nodes.size(); ++img_idx) {
        // get the relative path
        std::string img_relative_path =
            doc.get_node_attribute(img_nodes[img_idx], "path", "");
        if (img_relative_path.size() == 0) {
          printf("Img node #%i of person '%s' does not have 'path' "
                 "attribute set. Skipping img.\n", img_idx, name.c_str());
          continue;
        }
        // build the absolute path
        std::ostringstream img_absolute_path;
        img_absolute_path << faces_absolute_folder << img_relative_path;
        //printf("absolute_path:'%s'\n", absolute_path.str().c_str());
        // try to read the image
        PreprocessedBWFace curr_BW_img = cv::imread(img_absolute_path.str(),
                                                    CV_LOAD_IMAGE_GRAYSCALE);
        if (curr_BW_img.data == NULL) {
          printf("Img '%s', referred by node #%i of person '%s' "
                 "impossible to read. Skipping img.\n",
                 img_absolute_path.str().c_str(), img_idx, name.c_str());
          continue;
        }
        //  printf("absolute_path:'%s', curr_img:'%s'\n",
        //           absolute_path.str().c_str(),
        //           vision_utils::infosImage(curr_BW_img).c_str());

        preprocessed_faces.push_back(curr_BW_img.clone());
        names.push_back(name);
      } // end loop img_idx
    } // end loop person_idx

    // try to get model
    bool was_model_loaded = false;
    vision_utils::XmlDocument::Node* model_node = doc.get_node_at_direction(doc.root(), "model");
    if (model_node != NULL) {
      std::string model_filename = doc.get_node_attribute(model_node, "path");
      if (model_filename.size() > 0) {
        std::string model_full_filename = faces_absolute_folder + model_filename;
        try {
#if CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION >= 3 // from 3.3.1 onwards
          _model->load<cv_face::FisherFaceRecognizer>(model_full_filename);
#else
          _model->load(model_full_filename);
#endif
          printf("Model succesfully loaded from '%s'", model_full_filename.c_str());
          was_model_loaded = true;
        } catch (cv::Exception e) {
          printf("Error while loading model '%s':'%s', will retrain model.\n",
                 model_full_filename.c_str(), e.what());
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
    printf("to_xml_file('%s')\n", xml_absolute_filename.c_str());

    std::string faces_absolute_folder =
        vision_utils::extract_folder_from_full_path(xml_absolute_filename);

    // make folder
    std::ostringstream mkdir_order;
    mkdir_order << "mkdir " << faces_absolute_folder << " --parents";
    vision_utils::exec_system(mkdir_order.str());

    vision_utils::XmlDocument doc;
    // save model
#if 1
    std::string model_filename("model.yaml");
    _model->save(faces_absolute_folder + model_filename);
    vision_utils::XmlDocument::Node* model_node = doc.add_node(doc.root(), "model", "");
    doc.set_node_attribute(model_node, "path", model_filename);
#endif

    // make person nodes
    vision_utils::XmlDocument::Node* persons_node = doc.add_node(doc.root(), "persons", "");

    // get all existing names
    std::set<std::string> names;
    for (unsigned int label_idx = 0; label_idx < _labels.size(); ++label_idx) {
      PersonName name = person_label_to_name(_labels[label_idx]);
      if (name == "")
        continue;
      names.insert(name);
    } // end loop label

    // create corresponding nodes
    for (std::set<std::string>::iterator names_it = names.begin();
         names_it != names.end(); ++names_it) {
      // write the node
      vision_utils::XmlDocument::Node* person_node = doc.add_node(persons_node, "person", "");
      doc.set_node_attribute(person_node, "name", *names_it);
    }
    // doc.write_to_file("/tmp/foo.xml");

    // add all pictures to person nodes
    if(_BW_faces.size() != _labels.size()) {
      printf("%li _BW_faces != %li _labels\n", _BW_faces.size(), _labels.size());
      return false;
    }
    for (unsigned int img_BW_idx = 0; img_BW_idx < _BW_faces.size(); ++img_BW_idx) {
      PersonName name = person_label_to_name(_labels[img_BW_idx]);
      if (name == "")
        continue;
      // make full pathname
      std::ostringstream img_abs_path_str, img_rel_path_str;
      img_rel_path_str << name << "_" << img_BW_idx << ".png";
      img_abs_path_str << faces_absolute_folder << img_rel_path_str.str();
      // save image
      bool write_ok = cv::imwrite(img_abs_path_str.str(), _BW_faces[img_BW_idx]);
      if (!write_ok) {
        printf("Impossible to save face of person '%s' into file '%s'\n",
               name.c_str(), img_abs_path_str.str().c_str());
        return false;
      }
      // add attribute
      vision_utils::XmlDocument::Node* person_node =
          doc.get_node_at_direction(persons_node, "person", "name", name);
      if (person_node == NULL) {
        printf("Could not find 'person' node with name '%s'\n",
               name.c_str());
        continue;
      }
      vision_utils::XmlDocument::Node* img_node = doc.add_node(person_node, "img", "");
      doc.set_node_attribute(img_node, "path", img_rel_path_str.str());
    } // end loop img_BW

    doc.write_to_file(xml_absolute_filename);
    return true;
  } // end to_xml_file();

  //////////////////////////////////////////////////////////////////////////////

  PersonName predict_non_preprocessed_face
  (const NonPreprocessedColorFace & person_color_face) const {
    //    printf("predict_non_preprocessed_face(%s)\n",
    //           vision_utils::infosImage(person_color_face).c_str());
    PreprocessedBWFace person_BW_face;
    bool preprocess_success = preprocess_face(person_color_face, person_BW_face);
    if (!preprocess_success)
      return RECOGNITION_FAILED;
    return predict_preprocessed_face(person_BW_face);
  } // end predict_non_preprocessed_face();

  //////////////////////////////////////////////////////////////////////////////

  PersonName predict_color_image(const ColorImage & image) {
    //    printf("predict_color_image(%s)\n",
    //           vision_utils::infosImage(image).c_str());
    // vision_utils::Timer timer;
    cv::Mat3b small_img;
    std::vector< cv::Rect > found_faces;
    vision_utils::detect_with_opencv(image, classifier, small_img,
                                     found_faces);
    // Timer::Time t1 = timer.getTimeMilliseconds();
    // timer.reset();
    if (found_faces.size() == 0)
      return NOBODY;
    PersonName res = predict_non_preprocessed_face(image(found_faces.front()));
    //timer.printTime("predict_non_preprocessed_face()");
    //    printf("predict_color_image(): "
    //             "width, height, img size, t_detect_faces, t_predict_non_preprocessed_face  "
    //             "%i  %i  %i  %g  %g",
    //             image.cols, image.rows, image.cols * image.rows,
    //             t1, timer.getTimeMilliseconds());
    return res;
  }

  //////////////////////////////////////////////////////////////////////////////

  void predict_color_images(const std::vector<ColorImage> & images,
                            std::vector<PersonName> & results) {
    printf("predict_color_images(%li images)\n", images.size());
    results.resize(images.size());
    for (unsigned int img_idx = 0; img_idx < images.size(); ++img_idx) {
      if (img_idx % 100 == 0)
        printf("predict_color_images(%i images done / %li)",
               img_idx, images.size());
      results[img_idx] = predict_color_image(images[img_idx]);
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  void benchmark_color_images_filenames(const std::vector<std::string> & images_filenames,
                                        const std::vector<PersonName> & names) {
    printf("benchmark_color_images_filenames(%li images)",
           images_filenames.size());

    cv::Mat img;
    int success_nb = 0, failures_nb = 0, no_face_nb = 0;
    for (unsigned int img_idx = 0; img_idx < images_filenames.size(); ++img_idx) {
      if (img_idx % 100 == 0)
        printf("predict_color_images(%i images done / %li), "
               "success_nb:%i, failures_nb:%i, no_face_nb:%i",
               img_idx, images_filenames.size(),
               success_nb, failures_nb, no_face_nb);
      img = cv::imread(images_filenames[img_idx], CV_LOAD_IMAGE_COLOR);
      if (img.empty()) {
        printf("Error in reading image '%s', skipping it.\n",
               images_filenames[img_idx].c_str());
        continue;
      }
      PersonName result = predict_color_image(img);
      //printf("img %i ('%s') expected '%s', got '%s'\n",
      //  img_idx, images_filenames[img_idx].c_str(),
      //  names[img_idx].c_str(), result.c_str());
      if (result == NOBODY)
        ++no_face_nb;
      else if (result == names[img_idx])
        ++success_nb;
      else
        ++failures_nb;
    } // end loop img
    printf("success_nb:%i, failures_nb:%i, no_face_nb:%i\n",
           success_nb, failures_nb, no_face_nb);
  } // end benchmark_color_images_filenames();

  //////////////////////////////////////////////////////////////////////////////

  /*!
   Add a new face to a given person and retrain the model
   \param person_color_face
      the image to add
   \param name
     the person to be updated
   \return bool
      true if succesfully added,
      false otherwise (no face found in the image, preprocessing failed...)
  */
  bool add_non_preprocessed_face_to_person
  (const NonPreprocessedColorFace & person_color_face,
   const PersonName & name) {
    printf("add_non_preprocessed_face_to_person(%s: '%s')\n",
           vision_utils::infosImage(person_color_face).c_str(),
           name.c_str());
    PreprocessedBWFace person_BW_face;
    bool preprocess_success = preprocess_face(person_color_face, person_BW_face);
    if (!preprocess_success)
      return false;
    return add_preprocessed_face_to_person(person_BW_face, name);
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
    printf("from_preprocessed_faces(%li preprocessed faces, want_retrain_model:%i)",
           preprocessed_faces.size(), want_retrain_model);
    _BW_faces = preprocessed_faces; // copy faces
    _person_labels_map.clear();
    _labels.clear();
    _labels.reserve(preprocessed_faces.size());
    for (unsigned int face_idx = 0; face_idx < preprocessed_faces.size(); ++face_idx)
      _labels.push_back(name_to_person_label(names[face_idx]));

    if (want_retrain_model)
      retrain_model();
    return true;
  } // end from_preprocessed_faces()

  //////////////////////////////////////////////////////////////////////////////

  inline bool retrain_model() {
    printf("retrain_model(%li faces, %li labels '%s', _person_labels_map:'%s')\n",
           _BW_faces.size(), _labels.size(),
           vision_utils::accessible_to_string(_labels).c_str(),
           vision_utils::map_to_string(_person_labels_map).c_str());
    vision_utils::Timer timer;
    // check sizes
    if(_BW_faces.size() != _labels.size()) {
      printf("%li _BW_faces != %li _labels\n", _BW_faces.size(), _labels.size());
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
    //    vision_utils::paste_images(_BW_faces, out, 32, 32, 0, false);
    //    cv::imshow("out", out);
    //    cv::waitKey(0);
    //  for (unsigned int face_idx = 0; face_idx < _BW_faces.size(); ++face_idx)
    //    cv::imshow(vision_utils::cast_to_string(face_idx), _BW_faces[face_idx]);
    //  cv::waitKey(0);

    try {
      _model->train(_BW_faces, _labels);
      save_back_up_file();
    } catch (cv::Exception e) {
      printf("Exception while retrain_model():'%s'\n", e.what());
      return false;
    }
    printf("Time to train model:%f ms (%i people, %li faces in total)\n",
           timer.getTimeMilliseconds(), nb_persons_in_model(), _BW_faces.size());
    return true;
  } // end retrain_model();

  //////////////////////////////////////////////////////////////////////////////

  inline PersonName person_label_to_name(const PersonLabel & person_label) const {
    PersonName name;
    if (vision_utils::direct_search(_person_labels_map, person_label, name))
      return name;
    printf("Label #%i does not exist.\n", person_label);
    return "";
  } // end person_label_to_name()

  //////////////////////////////////////////////////////////////////////////////

  inline PersonLabel name_to_person_label
  (const PersonName & name) {
    PersonLabel person_label;
    bool lookup_success = vision_utils::reverse_search
        (_person_labels_map, name, person_label);
    if (lookup_success)
      return person_label;

    //    printf("Person '%s' is not a knwon person, _person_labels_map:'%s'. Adding it.\n",
    //             name.c_str(), vision_utils::map_to_string(_person_labels_map).c_str());
    // keep the association PersonLabel <-> PersonName
    person_label = _person_labels_map.size();
    _person_labels_map.insert
        (std::pair<PersonLabel, PersonName>(person_label, name));
    return person_label;
  } // end name_to_person_label()

  //////////////////////////////////////////////////////////////////////////////

  inline void save_back_up_file() const {
    std::ostringstream xml_filename_stream;
    xml_filename_stream << "/tmp/face_recognizer_backup_" << vision_utils::timestamp()
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
      vision_utils::resize_constrain_proportions(bw_face, bw_face, face_width, face_height);
      // equalize histogram
      cv::equalizeHist(bw_face, bw_face);
      // show image
      //cv::imshow("color_face", color_face); cv::imshow("bw_face", bw_face); cv::waitKey(1000);
    } catch (cv::Exception e) {
      printf("Error while preprocessing face:'%s'\n", e.what());
      return false;
    }
    return true;
  } // end preprocess_face()

  //////////////////////////////////////////////////////////////////////////////

  PersonName predict_preprocessed_face(const PreprocessedBWFace & face) const {
    //    printf("predict_preprocessed_face(%s)\n",
    //           vision_utils::infosImage(face).c_str());

    //PersonLabel person_label = _model->predict(face);
    double confidence;
    PersonLabel person_label;
    try {
      _model->predict(face, person_label, confidence);
    } catch (cv::Exception e) {
      printf("Error while predictiong with model:'%s'\n", e.what());
      return RECOGNITION_FAILED;
    }
    //printf("predict: person_label:%i, confidence:%g", person_label, confidence);

    PersonName name;
    bool lookup_success = vision_utils::direct_search
        (_person_labels_map, person_label, name);
    if (!lookup_success) {
      printf("The prediction is label #%i but it does not correspond to a "
             "knwon person, _person_labels_map:'%s'\n",
             person_label, vision_utils::map_to_string(_person_labels_map).c_str());
      return RECOGNITION_FAILED;
    }
    return name;
  }

  //////////////////////////////////////////////////////////////////////////////

  /*!
   \param person_BW_face
   \param name
   \return bool
  */
  bool add_preprocessed_face_to_person
  (const PreprocessedBWFace & person_BW_face,
   const PersonName & name)
  {
    printf("add_preprocessed_face_to_person(%s: '%s')\n",
           vision_utils::infosImage(person_BW_face).c_str(),
           name.c_str());

    // get the person label
    PersonLabel person_label = name_to_person_label(name);
    // add it to our data base
    _BW_faces.push_back(person_BW_face.clone());
    _labels.push_back(person_label);
    // re-train the model
    printf("Retraining the new model with this face.\n");
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
  cv::Ptr<cv_face::FaceRecognizer> _model;
}; // end class FaceRecognizer

} // end namespace face_recognition

#endif // FACE_RECOGNIZER_H
