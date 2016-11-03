/*!
  \file        person_histogram_set.h
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2012/12/12

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

#ifndef PERSON_HISTOGRAM_SET_H
#define PERSON_HISTOGRAM_SET_H

#include <opencv2/ml/ml.hpp>
#if CV_MAJOR_VERSION > 2
typedef cv::Ptr<cv::ml::SVM> MySVMPtr;
#else // OpenCV < 3.0
typedef cv::Ptr<cv::SVM> MySVMPtr;
#endif

#include "people_recognition_vision/person_histogram.h"

#include "vision_utils/colormaps.h"
#include "vision_utils/timestamp.h"
#include "vision_utils/array_to_color.h"
#include "cvstage/cvstage.h"


////////////////////////////////////////////////////////////////////////////////

class PersonHistogramSet {
public:
  typedef int PersonLabel;
  typedef CMatrix<double> CostMatrix;
  typedef vision_utils::MatchList PHMatch;
  typedef std::map<PersonLabel, PersonLabel> LabelMatch;

  static const int MAT_COLS = PersonHistogram::MAT_COLS;
  static const unsigned int CAPTION_WIDTH1 = 160, CAPTION_HEIGHT1 = 300, CAPTION_ITEMPADDING = 10;

  //////////////////////////////////////////////////////////////////////////////

  PersonHistogramSet() {
    dist_matrix_illus.create(1, 1);
    // draw caption
    col_titlemap = &vision_utils::int_to_uppercase_letter; // used for this PHS
    row_titlemap = &vision_utils::int_to_lowercase_letter; // used for other PHS
    colormap_to_caption_image(dist_matrix_colormap_caption, 100, 300,
                              vision_utils::ratio2red_green_half,
                              0., .6, .05, .1);
    refresh_unique_labels();
  } // end ctor

  //////////////////////////////////////////////////////////////////////////////

  /*!
   * template factory, that will make use of PersonHistogram::create()
   * it can even use vectors of vectors (creating structured histograms).
   */
  template<class _T1, class _T2, class _T3>
  bool push_back_vec(const std::vector<_T1> & v1,
                     const std::vector<_T2> & v2,
                     const std::vector<_T3> & v3,
                     const std::vector<PersonLabel> & labels,
                     bool want_refresh_illus_images = true,
                     bool retrain = true) {
    unsigned int new_hists_nb = v1.size();
    if (v2.size() != new_hists_nb
        || v3.size() != new_hists_nb
        || labels.size() != new_hists_nb) {
      printf("push_back_vec() sizes mismatch: v1:%i, v2:%i, v3:%i, labels:%i\n",
             v1.size(), v2.size(), v3.size(), labels.size());
      return false;
    }
    _phs.reserve(_phs.size() + new_hists_nb);
    for (unsigned int hist_idx = 0; hist_idx < new_hists_nb; ++hist_idx) {
      // do not retrain for intermediate images
      if (!push_back(v1[hist_idx], v2[hist_idx], v3[hist_idx], labels[hist_idx],
                     want_refresh_illus_images, false))
        return false;
    }
    return (retrain ? train_svm() : true);
  } // end ctor

  //  template<class _T1, class _T2, class _T3, class _T4>
  //  bool push_back_vec(const std::vector<_T1> & v1,
  //                     const std::vector<_T2> & v2,
  //                     const std::vector<_T3> & v3,
  //                     const std::vector<_T4> & v4,
  //                     const std::vector<PersonLabel> & labels,
  //                     bool want_refresh_illus_images = true,
  //                     bool retrain = true) {
  //    unsigned int new_hists_nb = v1.size();
  //    if (v2.size() != new_hists_nb
  //        || v3.size() != new_hists_nb
  //        || v4.size() != new_hists_nb
  //        || labels.size() != new_hists_nb) {
  //      printf("push_back_vec() sizes mismatch: v1:%i, v2:%i, v3:%i, v4:%i, labels:%i\n",
  //             v1.size(), v2.size(), v3.size(), v4.size(), labels.size());
  //      return false;
  //    }
  //    _phs.reserve(_phs.size() + new_hists_nb);
  //    for (unsigned int hist_idx = 0; hist_idx < new_hists_nb; ++hist_idx) {
  //      // do not retrain for intermediate images
  //      if (!push_back(v1[hist_idx], v2[hist_idx], v3[hist_idx], v4[hist_idx],
  //                     labels[hist_idx], want_refresh_illus_images, false))
  //        return false;
  //    }
  //    return (retrain ? train_svm() : true);
  //  } // end ctor

  //////////////////////////////////////////////////////////////////////////////

  template<class _T1, class _T2, class _T3>
  bool push_back(const _T1 & v1, const _T2 & v2, const _T3 & v3,
                 const PersonLabel & label,
                 bool want_refresh_illus_images = true,
                 bool retrain = true) {
    _phs.push_back(PersonHistogram());
    if (!_phs.back().create(v1, v2, v3, want_refresh_illus_images)) {
      _phs.pop_back(); // restore size
      return false;
    }
    _labels.push_back(label);
    refresh_unique_labels();
    return (retrain ? train_svm() : true);
  } // end ctor

  //////////////////////////////////////////////////////////////////////////////

  template<class _T1, class _T2, class _T3, class _T4>
  bool push_back(const _T1 & v1, const _T2 & v2, const _T3 & v3, const _T4 & v4,
                 const PersonLabel & label,
                 bool want_refresh_illus_images = true,
                 bool retrain = true) {
    _phs.push_back(PersonHistogram());
    if (!_phs.back().create(v1, v2, v3, v4, want_refresh_illus_images)) {
      _phs.pop_back(); // restore size
      return false;
    }
    _labels.push_back(label);
    refresh_unique_labels();
    return (retrain ? train_svm() : true);
  } // end ctor

  //////////////////////////////////////////////////////////////////////////////

  bool push_back(const PersonHistogram & ph, const PersonLabel & label, bool retrain = true) {
    _phs.push_back(ph);
    _labels.push_back(label);
    refresh_unique_labels();
    return (retrain ? train_svm() : true);
  } // end ctor

  //////////////////////////////////////////////////////////////////////////////

  bool pop_back(bool retrain = true) {
    _phs.pop_back();
    _labels.pop_back();
    refresh_unique_labels();
    return (retrain ? train_svm() : true);
  } // end ctor

  //////////////////////////////////////////////////////////////////////////////

  bool clear() {
    _phs.clear();
    _labels.clear();
    refresh_unique_labels();
    _SVM->clear();
    return true;
  } // end ctor

  //////////////////////////////////////////////////////////////////////////////

  bool erase_by_label(const PersonLabel & label) {
    for (int label_idx = 0; label_idx < (int) _labels.size(); ++label_idx) {
      if (_labels[label_idx] == label) {
        _labels.erase(_labels.begin() + label_idx);
        _phs.erase(_phs.begin() + label_idx);
        --label_idx;
      }
    } // end loop label_idx
    refresh_unique_labels();
    return train_svm();
  }

  //////////////////////////////////////////////////////////////////////////////

  bool to_mat(cv::Mat & out_mat) const {
    // create(rows, cols)
    out_mat.create(nhists(), MAT_COLS, CV_32FC1);
    for (unsigned int hist_idx = 0; hist_idx < nhists(); ++hist_idx) {
      cv::Mat out_row = out_mat.row(hist_idx);
      if (!_phs.at(hist_idx).to_mat(out_row)) {
        printf("to_mat() for hist %i failed\n", hist_idx);
        return false;
      }
    }
    return true;
  }

  //////////////////////////////////////////////////////////////////////////////

  inline const PersonHistogram & at(unsigned int i) const { return _phs.at(i); }

  inline bool last_at(PersonLabel lbl, const PersonHistogram* & out) const {
    for (int i = (int) _labels.size()-1; i >= 0; --i) {
      if (_labels[i] == lbl) {
        out = &_phs[i];
        return  true;
      }
    } // end loop i
    printf("last_at(): could not find label %i\n", lbl);
    out = NULL;
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////

  bool train_svm() {
    DEBUG_PRINT("train_svm(): %i hists, %i labels", nhists(), nlabels());

    if (nlabels() < 2) {
      printf("train_svm(): There are only %i labels\n", nlabels());
      return true;
    }

    // build training data
    if (!to_mat(_training_mat))
      return false;
    // find min and max values
    for (unsigned int i = 0; i < PersonHistogram::BODY_PARTS; ++i) {
      cv::minMaxIdx(_training_mat.col(2*i), &_mean_min[i], &_mean_max[i]);
      cv::minMaxIdx(_training_mat.col(2*i+1), &_std_dev_min[i], &_std_dev_max[i]);
    } // end loop i
    //std::cout << "_training_mat:" << _training_mat << std::endl;
    //printf("xmin:%g, xmax:%g, ymin:%g, ymax:%g\n", xmin, xmax, ymin, ymax);

    // generate labels_mat
    cv::Mat1f labels_mat;
    labels_mat.create(nhists(), 1) ;
    for (unsigned int label_idx = 0; label_idx < nhists(); ++label_idx)
      labels_mat.at<float>(label_idx) = _labels[label_idx];
    // rows, cols, data
    if (labels_mat.rows != (int) nhists() ||labels_mat.cols != 1) {
      printf("train_svm(): _labels_mat:'%s' != (1xnhists()=%i)\n",
             vision_utils::infosImage(labels_mat).c_str(), nhists());
      return false;
    }

#if CV_MAJOR_VERSION > 2 // OpenCV 3.0+
    // https://stackoverflow.com/questions/27114065/opencv-3-svm-training
    _SVM = cv::ml::SVM::create();
    // edit: the params struct got removed,
    // we use setter/getter now:
    _SVM->setType(cv::ml::SVM::C_SVC);
    _SVM->setKernel(cv::ml::SVM::RBF);
    _SVM->setC(10);
    _SVM->setGamma(5E-2);
    cv::TermCriteria crit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 1000, 1E-6);
    _SVM->setTermCriteria(crit);
    return (_SVM->train( labels_mat, cv::ml::ROW_SAMPLE , labels_mat ));
#else // OpenCV < 3.0
    // Set up SVM's parameters
    // from http://bytefish.de/blog/machine_learning_opencv/
    // cf also http://docs.opencv.org/modules/ml/doc/support_vector_machines.html#cvsvmparams
    CvSVMParams param;
    param.svm_type    = CvSVM::C_SVC;
    //param.kernel_type = CvSVM::LINEAR;
    param.kernel_type = CvSVM::RBF;
    param.C = 10; // the bigger, the bigger the penalty for misclassification
    param.gamma = 5E-2; // the smaller, the bigger the blobs
    //param.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    param.term_crit.type = CV_TERMCRIT_ITER +CV_TERMCRIT_EPS;
    param.term_crit.max_iter = 1000;
    param.term_crit.epsilon = 1e-6;
    // Train the SVM
    return _SVM->train(_training_mat, labels_mat, cv::Mat(), cv::Mat(), param);
    //return _SVM->train_auto(training_mat, _labels_mat, cv::Mat(), cv::Mat(), param);
#endif
  }

  //////////////////////////////////////////////////////////////////////////////

  cv::Mat3b svm2img(unsigned int body_part = 1) {
    if (body_part >= PersonHistogram::BODY_PARTS) {
      printf("svm2img(): incorrect body part %i, should be in [0, %i[\n",
             body_part, PersonHistogram::BODY_PARTS);
      return cv::Mat3b(10, 10, cv::Vec3b(0, 0, 0));
    }
    _ms.set_visible_window(_mean_min[body_part], _mean_max[body_part],
                           _std_dev_min[body_part], _std_dev_max[body_part]);
    std::string _window_name = "ms";
    cv::namedWindow(_window_name);
    _ms.set_mouse_move_callback(_window_name);
    while (true) {
      // paint response
#if 1
      unsigned int cols = _ms.cols(), rows = _ms.rows();
      cv::Mat query(1, MAT_COLS, CV_32FC1, cv::Scalar::all(0));
      for (unsigned int row = 0; row < rows; ++row) {
        cv::Vec3b* out_ptr = _ms.get_viz().ptr<cv::Vec3b>(row);
        for (unsigned int col = 0; col < cols; ++col) {
          // convert pixel to (mean, stddev)
          cv::Point2f meand_stddev = _ms.pixel2world(col, row);
          query.at<float>(0, 2*body_part  ) = meand_stddev.x;
          query.at<float>(0, 2*body_part+1) = meand_stddev.y;
          // use the SVM
          int label = _SVM->predict(query);
          //    if (rand() % 100000 == 0)
          //      printf("mean:%g, std_dev:%g, label:%i\n",
          //             meand_stddev.x, meand_stddev.y, label);
          // paint it
          out_ptr[col] = vision_utils::color<cv::Vec3b>(label);
        } // end loop col
      } // end loop row
#else
      ms.clear();
#endif
      for (int training_idx = 0; training_idx < _training_mat.rows; ++training_idx) {
        cv::Point sample = _ms.world2pixel(_training_mat.at<float>(training_idx, 0),
                                           _training_mat.at<float>(training_idx, 1));
        vision_utils::drawCross(_ms.get_viz(), sample, 3, CV_RGB(0, 0, 0), 2);
      } // end loop training_idx

      _ms.draw_grid(5, 150);
      _ms.draw_axes();

      cv::imshow(_window_name, _ms.get_viz());
      char c = cv::waitKey(50);
      if ((int) c == 27)
        break;
    } // end while (true)
    return _ms.get_viz();
  } // end svm2img()

  //////////////////////////////////////////////////////////////////////////////

  bool compare_svm(const PersonHistogram & ph,
                   int & out_label) {
    cv::Mat ph_mat(1, MAT_COLS, CV_32FC1);
    if (!ph.to_mat(ph_mat))
      return false;
    float out_label_float = _SVM->predict(ph_mat);
    // printf("out_label_float:%g\n", out_label_float);
    out_label = out_label_float;
    return true;
  }

  //////////////////////////////////////////////////////////////////////////////

  void generate_caption_image
  (cv::Mat3b & caption_illus,
   vision_utils::Map row_titlemap = &vision_utils::int_to_lowercase_letter)
  const {
    uint nimgs = nlabels();
    if (nimgs == 0) { // check if histogram empty
      caption_illus.create(1, 1);
      caption_illus.setTo(255);
      return;
    }

    std::vector<cv::Mat3b> illus_color_imgs;
    std::vector<cv::Mat1b> illus_mask_imgs;
    illus_color_imgs.reserve(nimgs);
    illus_mask_imgs.reserve(nimgs);
    for (unsigned int label_idx = 0; label_idx < nlabels(); ++label_idx) {
      PersonHistogramSet::PersonLabel curr_label = _unique_labels[label_idx];
      const PersonHistogram* ph = NULL;
      if (!last_at(curr_label, ph))
        continue;
      illus_color_imgs.push_back(ph->get_illus_color_img());
      // printf("color:'%s'\n", vision_utils::infosImage(illus_color_imgs.back()).c_str());
      illus_mask_imgs.push_back(ph->get_illus_color_mask());
      // printf("mask:'%s'\n", vision_utils::infosImage(illus_mask_imgs.back()).c_str());
    } // end loop label_idx
    vision_utils::paste_images(illus_color_imgs, caption_illus,
                              true, CAPTION_WIDTH1, CAPTION_HEIGHT1, CAPTION_ITEMPADDING,
                              true, row_titlemap, illus_mask_imgs, true);

  } // end generate_caption_image();

  ////////////////////////////////////////////////////////////////////////////////

  /*!
   * Compare to a bunch of other histograms
   * \param hists_new
   * \param generate_illus
   * \param method
   * \param are_vector_equals
   * \return the index of the most similar
   */
  bool compare_to(const PersonHistogramSet & hists2,
                  PHMatch & best_assign,
                  bool generate_dist_matrix_illus = false,
                  bool generate_dist_matrix_illus_caption = false,
                  const int method = CV_COMP_BHATTACHARYYA,
                  bool are_vector_equals = false) {
    DEBUG_PRINT("compare_to()");
    // compute cost matrix:
    uint nhists1 = nhists(), nhists2 = hists2.nhists();
    //printf("compare_to(): %i hists, h2: %i hists\n", nhists1, nhists2);
    if (are_vector_equals) {
      // it is a symmetric matrix
      dist_matrix.resize(nhists1, nhists1);
      for (unsigned int row = 0; row < nhists1; ++row) {
        dist_matrix[row][row] = 0; // same histogram -> distance of zero
        for (unsigned int col = row + 1; col < nhists1; ++col)
          dist_matrix[row][col] =
              dist_matrix[col][row] = _phs[row].compare_to(_phs[col], method);
      } // end loop row
    } // end if (are_vector_equals)
    else { // are_vector_equals false
      dist_matrix.resize(nhists2, nhists1);
      for (unsigned int row = 0; row < nhists2; ++row) {
        for (unsigned int col = 0; col < nhists1; ++col) {
          dist_matrix[row][col] = hists2._phs[row].compare_to(_phs[col], method);
          // printf("dist_matrix[%i][%i]:%g\n", row, col, dist_matrix[row][col]);
        }
      } // end loop row
    } // end not are_vector_equals
    // ROS_WARN("dist_matrix:'%s'", dist_matrix.to_string(15).c_str());

    // make linear assignment
    best_assign.clear();
    vision_utils::Cost best_cost;
    bool assign_success = vision_utils::linear_assign
        (dist_matrix, best_assign, best_cost);
    if (!assign_success)
      return false;
    // reverse it, as rows correspond to hists2
    vision_utils::reverse_assignment_list(best_assign);

    if (generate_dist_matrix_illus) {
      DEBUG_PRINT("compare_person_histograms_generate_images()");
      // build custom headers: rows = hists1, cols = hists2
      row_titlemap = (are_vector_equals ? &vision_utils::int_to_uppercase_letter : &vision_utils::int_to_lowercase_letter);
      std::vector<std::string> col_headers, row_headers;
      generate_headers(col_titlemap, col_headers);
      if (are_vector_equals)
        col_headers = row_headers;
      else
        hists2.generate_headers(row_titlemap, row_headers);

      // draw distance array with no edges and red_green mmap
      array_to_color(dist_matrix, nhists2, nhists1,
                     dist_matrix_illus, 60, 40, false, true,
                     vision_utils::ratio2red_green_half,
                     col_titlemap, row_titlemap, &row_headers, &col_headers);
    }

    if (generate_dist_matrix_illus_caption) {
      // caption for hists1
      generate_caption_image(dist_matrix_illus_caption1, row_titlemap);
      if (are_vector_equals)
        dist_matrix_illus_caption1.copyTo(dist_matrix_illus_caption2);
      else
        // caption for hists2
        hists2.generate_caption_image(dist_matrix_illus_caption2, col_titlemap);
    } // end if (generate_illus)

    return true;
  }

  ////////////////////////////////////////////////////////////////////////////////

  bool compare_to(const PersonHistogram & ph,
                  int & out_label,
                  bool generate_dist_matrix_illus = false,
                  bool generate_dist_matrix_illus_caption = false,
                  const int method = CV_COMP_BHATTACHARYYA) {
    out_label = -1;
    int hist_pos = 0;
    PersonHistogramSet phs;
    if (!phs.push_back(ph, hist_pos, false)) { // do not train SVM, not needed
      printf("compare_to(): impossible to create a new PersonHistogramSet\n");
      return false;
    }
    PHMatch assign;
    if (!compare_to(phs, assign, generate_dist_matrix_illus, generate_dist_matrix_illus_caption, method))
      return false;
    unsigned int nassigns = assign.size();
    for (unsigned int assign_idx = 0; assign_idx < nassigns; ++assign_idx) {
      if (assign[assign_idx].second == hist_pos) {
        int ph_idx = assign[assign_idx].first;
        out_label = label(ph_idx);
        if (out_label == -1) {
          printf("compare_to(): the new person histogram was not assigned:'%s'\n",
                 vision_utils::assignment_list_to_string(assign).c_str());
          return false;
        }
        return true;
      }
    } // end for assign_idx
    printf("compare_to(): could not find index 0 in assign:'%s'\n",
           vision_utils::assignment_list_to_string(assign).c_str());
    return false;
  } // end compare_to()

  //////////////////////////////////////////////////////////////////////////////

  bool assign2labels(const PHMatch & best_assign, LabelMatch & out) const {
    out.clear();
    for (unsigned int assign_idx = 0; assign_idx < best_assign.size(); ++assign_idx) {
      const vision_utils::Match* asg = &best_assign[assign_idx];
      if (asg->first == vision_utils::UNASSIGNED
          || asg->second == vision_utils::UNASSIGNED)
        continue;
      PersonLabel l1 = label(asg->first), l2 = label(asg->second);
      if (l1 == -1 || l2 == -1)
        continue;
      out[l1] = l2;
    } // end for (assign_idx)
    return true;
  }

  //////////////////////////////////////////////////////////////////////////////

  inline PersonLabel label(const unsigned int idx) const {
    if (idx >= nhists()) {
      printf("label():idx=%i >= nhists()=%i\n", idx, nhists());
      return -1;
    }
    if (idx >= _labels.size()) {
      printf("label():idx=%i >= _labels_mat.rows=%i\n", idx, _labels.size());
      return -1;
    }
    return _labels[idx];
  }

  //////////////////////////////////////////////////////////////////////////////

  inline bool to_yaml(const std::string & full_filename) const {
    ROS_INFO("to_yaml('%s')", full_filename.c_str());
    //vision_utils::to_yaml_vector(_phs, yaml_filename_prefix, "PersonHistogramSet");
    cv::FileStorage fs(full_filename, cv::FileStorage::WRITE);
    if (!fs.isOpened())
      return false;
    vision_utils::write(_phs, fs, "phs");
    //fs << "SVM" << _SVM;
    //_SVM->write(&fs, "SVM");
    fs << "labels" << _labels;
    fs << "training_mat" << _training_mat;
    fs.release();
    return true;
  }

  //////////////////////////////////////////////////////////////////////////////

  inline bool from_yaml(const std::string & full_filename) {
    ROS_INFO("from_yaml('%s')", full_filename.c_str());
    //vision_utils::from_yaml_vector(_phs, yaml_filename_prefix, "PersonHistogramSet");
    cv::FileStorage fs(full_filename, cv::FileStorage::READ);
    if (!fs.isOpened())
      return false;
    fs["phs"] >> _phs;
    //fs["SVM"] >> _SVM;
    fs["labels"] >> _labels;
    fs["training_mat"] >> _training_mat;
    fs.release();
    refresh_unique_labels();
    return train_svm();
  }

  //////////////////////////////////////////////////////////////////////////////

  inline void save_back_up_file() const {
    std::ostringstream xml_filename_stream;
    xml_filename_stream << "/tmp/PersonHistogramSet_backup_" << vision_utils::timestamp();
    to_yaml(xml_filename_stream.str());
  }


  //////////////////////////////////////////////////////////////////////////////

  /*! \return A1, A2, B1
   *  the letter identifies the label
   *  and the number the index of the Ph with this label
   */
  inline bool generate_headers(vision_utils::Map titlemap,
                               std::vector<std::string> & ans) const {
    ans.clear();
    ans.reserve(nhists());
    std::map<PersonLabel, std::string> labels2letter;
    std::vector<PersonLabel> labels = get_unique_labels();
    for (unsigned int label_idx = 0; label_idx < nlabels(); ++label_idx)
      labels2letter.insert(std::make_pair(labels[label_idx], titlemap(label_idx)));

    std::map<PersonLabel, int> counts;
    for (unsigned int hist_idx = 0; hist_idx < nhists(); ++hist_idx) {
      PersonLabel curr_label = label(hist_idx);
      counts[curr_label] = counts[curr_label]+1;
      std::string header = labels2letter[curr_label] + vision_utils::cast_to_string(counts[curr_label]);
      ans.push_back(header);
    } // end loop hist_idx
    return true;
  }

  //////////////////////////////////////////////////////////////////////////////

  inline unsigned int nhists() const { return _phs.size(); }
  inline unsigned int nlabels() const { return _nlabels; }
  inline unsigned int maxlabel() const { return (_nlabels == 0 ? 0 : _unique_labels.back()); }
  inline std::vector<PersonLabel> get_unique_labels() const { return _unique_labels; }

  inline const CostMatrix& get_dist_matrix() const { return dist_matrix; }
  inline const cv::Mat3b& get_dist_matrix_illus() const { return dist_matrix_illus; }
  inline const cv::Mat3b& get_dist_matrix_illus_caption1() const { return dist_matrix_illus_caption1; }
  inline const cv::Mat3b& get_dist_matrix_illus_caption2() const { return dist_matrix_illus_caption2; }
  inline const cv::Mat3b& get_dist_matrix_colormap_caption() const { return dist_matrix_colormap_caption; }

private:

  //////////////////////////////////////////////////////////////////////////////

  inline void refresh_unique_labels() {
    std::set<float> unique_labels_set;
    for (unsigned int label_idx = 0; label_idx < _labels.size(); ++label_idx)
      unique_labels_set.insert(_labels[label_idx]);
    _nlabels = unique_labels_set.size();
    _unique_labels.clear();
    for (std::set<float>::const_iterator l = unique_labels_set.begin(); l != unique_labels_set.end(); ++l)
      _unique_labels.push_back((PersonLabel) *l);
  }

  //////////////////////////////////////////////////////////////////////////////

  std::vector<PersonHistogram> _phs;

  // data
  CostMatrix dist_matrix;

  // viz
  vision_utils::Map col_titlemap, row_titlemap; // col: this, row: other
  cv::Mat3b dist_matrix_illus_caption1, dist_matrix_illus_caption2;
  cv::Mat3b dist_matrix_colormap_caption;
  cv::Mat3b dist_matrix_illus;

  int _nlabels;
  std::vector<PersonLabel> _labels;
  std::vector<PersonLabel> _unique_labels; // _labels without repetition
  cv::Mat1f _training_mat;
  MySVMPtr _SVM;

  // viz stuff
  MiniStage _ms;
  double _mean_min[MAT_COLS], _mean_max[MAT_COLS],
  _std_dev_min[MAT_COLS], _std_dev_max[MAT_COLS];
}; // end class PersonHistogramSet

////////////////////////////////////////////////////////////////////////////////

#endif // PERSON_HISTOGRAM_SET_H
