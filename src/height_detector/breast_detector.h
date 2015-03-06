/*!
  \file        breast_detector.h
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

A class for determining the gender of a user
thanks to the size of his/her breast.
 */

#ifndef BREAST_DETECTOR_H
#define BREAST_DETECTOR_H

#include "height_detector/height_detector.h"
#include <point_clouds/cloud_tilter.h>
#include <visu_utils/mini_stage_plugins.h>
#include <geom/Rect3.h>
#include <string/find_and_replace.h>
#include <string/string_split.h>
//#define DEBUG
#include <debug/debug2.h>
// opencv
#include <opencv2/ml/ml.hpp>
// ros
#include <ros/package.h>

#if 0
#define TIMER_CREATE(t)                     Timer t;
#define TIMER_RESET(t)                      t.reset();
#define TIMER_PRINT_RESET(t, msg)           t.printTime(msg); t.reset();

#else // do nothing
#define TIMER_CREATE(t)                     // empty
#define TIMER_RESET(t)                      // empty
#define TIMER_PRINT_RESET(t, msg)           // empty
#endif // CHART_TIMER_ON

//#define USE_PCL_VIEWER
#ifdef USE_PCL_VIEWER
//#include <point_clouds/cloud_viewer.h>
#else
#include <point_clouds/cloud_viewer_gnuplot.h>
#endif // no USE_PCL_VIEWER

class IndexDoubleConverter {
public:
  IndexDoubleConverter() {}
  IndexDoubleConverter(const double & min, const double & max, const int & ncells) {
    _min = min;
    _max = max;
    _ncells = ncells;
    _index2double = 1. * (_max - _min) / _ncells;
    _double2index = 1. * _ncells / (_max - _min);
  }
  inline double index2double(const int index) const {
    return _min + index * _index2double;
  }
  inline int double2index(const double d) const {
    return (d - _min) * _double2index;
  }

  double _min, _max;
  int _ncells;
  // cached computation
  double _index2double, _double2index;
};

////////////////////////////////////////////////////////////////////////////////

class BreastDetector {
public:
  typedef cv::Point   Pt2i;
  typedef cv::Point2f Pt2f;
  typedef cv::Point3f Pt3f;
  static const bool KEEP_RATIO = true;
  //////////////////////////////////////////////////////////////////////////////
  enum Gender {
    NOT_COMPUTED = HeightDetector::NOT_COMPUTED,
    ERROR = HeightDetector::ERROR,
    MALE = 0,
    FEMALE = 1
  };
  //////////////////////////////////////////////////////////////////////////////
  struct HeightBreast : HeightDetector::Height {
    Gender gender;
    double gender_confidence;
    HeightBreast()
      : gender(NOT_COMPUTED),
        gender_confidence (NOT_COMPUTED) {}
  };
  //////////////////////////////////////////////////////////////////////////////
  enum Method {
    WALK3D = 0,
    REPROJECT = 1,
    TEMPLATE_MATCHING = 2
  };
  //////////////////////////////////////////////////////////////////////////////
  struct Template {
    Template(const double & a_, const double & b_, const double & s_) : a(a_), b(b_), s(s_) {}
    double a, b, s;
    std::vector<BreastDetector::Pt2f> pts;
  };
  //////////////////////////////////////////////////////////////////////////////
  enum SvmStatus {
    SVM_STATUS_NOT_LOADED = 0,
    SVM_STATUS_TRAINED_SUCCESFULLY = 1,
    SVM_STATUS_LOADED_FAILED = 2
  };
  //////////////////////////////////////////////////////////////////////////////

  static const char* SVM_FILE(Method method) {
    if (method == WALK3D)
      return IMG_DIR "breast/BreastDetector_walk3d_svm.yaml";
    else if (method == REPROJECT)
      return IMG_DIR "breast/BreastDetector_reproject_svm.yaml";
    else if (method == TEMPLATE_MATCHING)
      return IMG_DIR "breast/BreastDetector_template_matching_svm.yaml";
    else return "/dev/null";
  }

  //////////////////////////////////////////////////////////////////////////////

  //! ctor
  BreastDetector(bool display_svm = false)
    : _template_converter(-.25/*TEMPLATE_xmin*/, .25/*TEMPLATE_xmax*/, TEMPLATE_NPTS) {
    _best_slice_idx = 0;
    // WALK3D
    std::vector<std::string> models;
    if (KEEP_RATIO) {
      models.push_back(IMG_DIR "breast/man_breast_model.png");
      models.push_back(IMG_DIR "breast/woman_breast_model.png");
    } else {
      models.push_back(IMG_DIR "breast/man_breast_model_stretched.png");
      models.push_back(IMG_DIR "breast/woman_breast_model_stretched.png");
    }
    _breast_comparer.set_models(models, cv::Size(32, 32), KEEP_RATIO);
    if (!_breast_comparer.get_models_nb() == 2) {
      printf("Could not correctly load breast models '%s', '%s'\n",
             models[0].c_str(), models[1].c_str());
      exit(-1);
    }
    _svm_walk_3d_status = SVM_STATUS_NOT_LOADED;
    _svm_reproject_status = SVM_STATUS_NOT_LOADED;
    _svm_template_matching_status = SVM_STATUS_NOT_LOADED;
  }

  //////////////////////////////////////////////////////////////////////////////

  inline HeightBreast detect_breast(const cv::Mat1f & depth,
                                    const cv::Mat1b & user_mask,
                                    const image_geometry::PinholeCameraModel & depth_cam_model,
                                    Method method = WALK3D) {
    if (method == WALK3D)
      return detect_breast_walk3d(depth, user_mask, depth_cam_model);
    else if (method == REPROJECT)
      return detect_breast_reproject(depth, user_mask, depth_cam_model);
    else if (method == TEMPLATE_MATCHING)
      return detect_breast_template_matching(depth, user_mask, depth_cam_model);
    else {
      printf("detect_breast(): Unknown method %i\n", method);
      HeightBreast ans;
      ans.gender = ERROR;
      return ans;
    }
  }

  //////////////////////////////////////////////////////////////////////////////

  bool train(const std::vector<std::string> & rgb_depth_user_filename_prefixes,
             const std::vector<uchar> & user_indices,
             const std::vector<image_geometry::PinholeCameraModel> & depth_cam_models,
             const std::vector<Gender> & labels,
             Method method = WALK3D) {
    unsigned int nsamples = rgb_depth_user_filename_prefixes.size();
    if (user_indices.size() != nsamples || depth_cam_models.size() != nsamples
        || labels.size() != nsamples) {
      printf("train(): non consistent size in the training data! "
             "%i filenames, %i user indces, %i cam models, %i labels\n",
             nsamples, user_indices.size(), depth_cam_models.size(), labels.size());
      return false;
    }
    std::vector< std::vector<double> > features_ok;
    std::vector< int > labels_ok;
    features_ok.reserve(nsamples);
    labels_ok.reserve(nsamples);
    int feature_size = -1;

    for (unsigned int sample_idx = 0; sample_idx < nsamples; ++sample_idx) {
      if (sample_idx % 100 == 0)
        printf("train():  --> processed file %i of %i\n", sample_idx, nsamples);
      // load the files
      std::string filename = rgb_depth_user_filename_prefixes[sample_idx];
      cv::Mat1b user_mask;
      cv::Mat3b rgb;
      cv::Mat1f depth;
      if (!image_utils::read_rgb_depth_user_image_from_image_file
          (filename, &rgb, &depth, &user_mask))
        return false;
      user_mask = (user_mask == user_indices[sample_idx]);
      // call detect_breast() - output does not matter
      _feature.clear();
      detect_breast(depth, user_mask, depth_cam_models[sample_idx], method);
      // display
      //cv::Mat3b breast_illus;
      //breast2img(user_mask, breast_illus, HeightBreast(), method);
      // cv::imshow("breast_illus", breast_illus); cv::waitKey(5);

      // check size of featyre
      if (_feature.empty()) {
        printf("train(): '%s': feature empty, detect_breast() must have failed!\n",
               filename.c_str());
        //cv::waitKey(0);
        continue;
      }
      if (feature_size == -1) // store the feature size
        feature_size= _feature.size();
      if ((int) _feature.size() != feature_size) {
        printf("train(): '%s': Feature size %i != the correct size %i\n",
               filename.c_str(), _feature.size(), feature_size);
        // cv::waitKey(0);
        continue;
      }
      // keep data
      features_ok.push_back(_feature);
      labels_ok.push_back(labels[sample_idx]);
    } // end loop sample_idx

    // now turn to matrices
    unsigned int nsamples_ok = labels_ok.size(); // rows, cols
    cv::Mat1f training_mat(nsamples_ok, feature_size), labels_mat(nsamples_ok, 1);
    // keep the feature in training_mat
    for (unsigned int sample_idx = 0; sample_idx < nsamples_ok; ++sample_idx) {
      for (int i = 0; i < feature_size; ++i)
        training_mat.at<float>(sample_idx, i) = features_ok[sample_idx][i]; // row, col
      labels_mat.at<float>(sample_idx) = labels_ok[sample_idx];
    }
    if (!train_svm(method, training_mat, labels_mat))
      return false;

    return save_svm(method, training_mat, labels_mat);
  } // end train()

  //////////////////////////////////////////////////////////////////////////////

  //! display with GNUplot - http://dirsig.blogspot.com.es/2010/11/lidar-point-cloud-visualization.html
  void display_svm_samples(const cv::Mat1f & training_mat,
                           const cv::Mat1f & labels_mat) {
    int nsamples = training_mat.rows;
    std::ostringstream data, instr;
    for (int sample_idx = 0; sample_idx < nsamples; ++sample_idx)
      data << training_mat.at<float>(sample_idx, 0) << ' '
           << training_mat.at<float>(sample_idx, 1) << ' '
           << training_mat.at<float>(sample_idx, 2) << ' '
           << labels_mat.at<float>(sample_idx) << std::endl;
    StringUtils::save_file("/tmp/BreastDetector.data", data.str());
    instr << "gnuplot -e \""
          << "set palette rgb 33,13,10; "
          << "splot '/tmp/BreastDetector.data' using 1:2:3:4 with points palette title 'Gender'; "
          << "pause -1 \"";
    system_utils::exec_system(instr.str());
  }

  //////////////////////////////////////////////////////////////////////////////

  bool breast_all_values(const cv::Mat1f & depth,
                         const cv::Mat1b & user_mask,
                         const image_geometry::PinholeCameraModel & depth_cam_model,
                         std::map<int, HeightBreast> & heights,
                         Method method = WALK3D,
                         bool want_illus = false,
                         cv::Mat3b* breasts_illus = NULL) {
    if (user_mask.empty() || depth.size() != user_mask.size()) {
      printf("height_meters_all_values: dimensions of depth(%i, %i), "
             "user_mask(%i, %i) dont match!\n",
             depth.cols, depth.rows, user_mask.cols, user_mask.rows);
      return false;
    }
    heights.clear();
    if (want_illus) {
      breasts_illus->create(depth.size());
      breasts_illus->setTo(0);
    }
    // find all users
    std::vector<uchar> user_indices;
    image_utils::get_all_different_values(user_mask, user_indices, true);
    unsigned int nusers = user_indices.size();

    // iterate on all user values
    cv::Mat1b curr_user_mask;
    for (unsigned int user = 0; user < nusers; ++user) {
      curr_user_mask = (user_mask == user_indices[user]);
      HeightBreast curr_height = detect_breast(depth, curr_user_mask, depth_cam_model, method);
      heights.insert(std::pair<int, HeightBreast>(user_indices[user], curr_height));

      if (want_illus && curr_height.gender_confidence != ERROR) // only draw if success
        breast2img(curr_user_mask, *breasts_illus, curr_height, method);
    } // end loop user
    return true;
  } // end  breast_meters_all_values()

  //////////////////////////////////////////////////////////////////////////////

  inline bool breast2img(const cv::Mat1b & user_mask,
                         cv::Mat3b & breast_illus,
                         HeightBreast h,
                         Method method) {
    if (method == WALK3D)
      return breast2img_walk3d(user_mask, breast_illus, h);
    else if (method == REPROJECT)
      return breast2img_reproject(user_mask, breast_illus, h);
    else if (method == TEMPLATE_MATCHING)
      return breast2img_template_matching(user_mask, breast_illus, h);
    printf("breast2img(): Unknown method %i\n", method);
    breast_illus.create(100, 100);
    breast_illus.setTo(0);
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////

  void illus_pcl(const cv::Mat1f & depth,
                 const cv::Mat1b & user_mask,
                 const image_geometry::PinholeCameraModel & depth_cam_model,
                 Method method) {
#ifdef USE_PCL_VIEWER
    // reproject user
    cloud_viewer::Viewer viewer("SimpleCloudViewer");
    viewer.setBackgroundColor (0, 0, 0);
    viewer.addCoordinateSystem (1.0);
    viewer.initCameraParameters ();
    // x *            O
    //    \         ==|==
    //     +--* z     |
    //     |         | |
    //   y *         | |
    Pt3f centroid;

    if (method == WALK3D) {
      // reprojected user_3D
      if (!reproject_user_to_3D(depth, user_mask, depth_cam_model, false))
        return;
      // viewer.addPointCloud(cloud_viewer::cloud2pcl(_path3D), "path3D");
      _head2feet_path_colors.resize(_head2feet_path3D.size(), cv::Vec3b(0, 255, 0));

      viewer.addPointCloud(cloud_viewer::rgb_cloud2pcl(_head2feet_path3D, _head2feet_path_colors),
                           "path3D");
      centroid = _breast_bbox3D.centroid<Pt3f>();
    } // end if (method == WALK3D)
    else { // REPROJECT
      geometry_utils::Rect3_<float> _user_bbox3D =
          geometry_utils::boundingBox_vec3<float, Pt3f, std::vector<Pt3f> >(_user_3D);
      centroid = _user_bbox3D.centroid<Pt3f>();
    }
    // all methods have reprojected _user_3D
    viewer.addPointCloud(cloud_viewer::cloud2pcl(_user_3D), "user3D");

    printf("centroid:'%s'\n", geometry_utils::printP(centroid).c_str());
    cloud_viewer::look_at(viewer, -0.1, 0, 0.1,
                          centroid.x, centroid.y, centroid.z,    0, -1, 0);
    viewer.spin();

    // show breast
    if (method == WALK3D) {
      _breast_path_colors.resize(_breast_path3D.size(), cv::Vec3b(0, 0, 255));
      viewer.removeAllPointClouds();
      viewer.addPointCloud(cloud_viewer::rgb_cloud2pcl(_breast_path3D, _breast_path_colors),
                           "breast_path3D");
      viewer.addCube(_breast_bbox3D.x, _breast_bbox3D.x + _breast_bbox3D.width,
                     _breast_bbox3D.y, _breast_bbox3D.y + _breast_bbox3D.height,
                     _breast_bbox3D.z, _breast_bbox3D.z + _breast_bbox3D.depth);
      viewer.spin();
    } // end if (method == WALK3D)
#else
    CloudViewerGnuPlot viewer;
    //  if (_user_3D.size() > 0)
    //    viewer.view_cloud(_user_3D, "user_3D");
    if (_head2feet_path3D.size() > 0)
      viewer.view_cloud(_head2feet_path3D, "head2feet_path3D");
#endif // no USE_PCL_VIEWER
  } // end illus_cv

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

protected:

  //////////////////////////////////////////////////////////////////////////////

  bool fit_ellipse_and_rotate(Pt3f & translation, double & rotation) {
    if (_user_3D.size() < 1) {
      printf("fit_ellipse_and_rotate(): _user_3D empty!\n");
      return false;
    }
    // now project all these points on the ground (xz), so remove y
    _user_ymin = _user_3D[0].y;
    _user_ymax = _user_ymin;
    unsigned int npts = _user_3D.size();
    _proj_x.resize(npts);
    _proj_y.resize(npts);
    for (unsigned int pt_idx = 0; pt_idx < npts; ++pt_idx) {
      _proj_x[pt_idx] = _user_3D[pt_idx].x;
      _proj_y[pt_idx] = _user_3D[pt_idx].z;
      // refresh bbox in y
      double curr_y = _user_3D[pt_idx].y;
      if (_user_ymin > curr_y)
        _user_ymin = curr_y;
      else if (_user_ymax < curr_y)
        _user_ymax = curr_y;
    } // end loop pt_idx
    _user_height = _user_ymax - _user_ymin;

    if (!gaussian_pdf_ellipse(_proj_x, _proj_y,
                              _ellipse_center, _ellipse_end1, _ellipse_end2, 2)) {
      printf("fit_ellipse_and_rotate(): gaussian_pdf_ellipse() failed!\n");
      return false;
    }
    _ellipse = ellipse_utils::three_pts2ellipse(_ellipse_center, _ellipse_end1, _ellipse_end2);

    // find the end of the short axis of the ellipse closest to origin
    Pt2f long1, long2, short1, short2;
    ellipse_utils::ellipse_axes(_ellipse, long1, long2, short1, short2);
    _ellipse_closest_end = short1;
    if (geometry_utils::distance_points_squared(short1, Pt2f(0, 0))
        > geometry_utils::distance_points_squared(short2, Pt2f(0, 0)))
      _ellipse_closest_end = short2;

    // center the point cloud in 0, looking towards x
    translation = Pt3f(-_ellipse_center.x, -_ellipse_center.y, 0);
    rotation = -atan2(_ellipse_closest_end.y - _ellipse_center.y,
                      _ellipse_closest_end.x - _ellipse_center.x);
    return true;
  }

  //////////////////////////////////////////////////////////////////////////////

  inline bool reproject_user_to_3D
  (const cv::Mat1f & depth,
   const cv::Mat1b & user_mask,
   const image_geometry::PinholeCameraModel & depth_cam_model,
   bool erode = false,
   bool straighten = true)
  {
    if (erode) {
      cv::Rect bbox = image_utils::boundingBox(user_mask);
      if (bbox.x < 0) {
        printf("HeightBreast: input user_mask '%s 'empty!\n",
               image_utils::infosImage(user_mask).c_str());
        return false;
      }
      int erode_kernel_size = clamp(bbox.width / 8, 10, 30);
      debugPrintf("erode_kernel_size:%i\n", erode_kernel_size);
      cv::Mat erode_kernel = cv::Mat(erode_kernel_size, erode_kernel_size, CV_8U, 255);
      cv::erode(user_mask, user_mask_eroded, erode_kernel);
    }

    // reproject user in 3D
    if (!kinect_openni_utils::pixel2world_depth
        (depth, depth_cam_model, _user_3D, 1,
         (erode ? user_mask_eroded : user_mask), true))
      return false;
    // straighten point cloud if needed
    if (straighten)
      return _tilter.straighten(_user_3D);
    return true;
  }

  //////////////////////////////////////////////////////////////////////////////

  /*!
   * Needs the _head2feet_path3D to contain the point cloud of the user from head to feet
   */
  inline HeightBreast convert_head2feet_path_to_feature(Method method) {
    HeightBreast herror;
    herror.gender_confidence = herror.gender = ERROR;
    if (_head2feet_path3D.empty()) {
      printf("convert_head2feet_path_to_feature(): Empty _head2feet_path3D\n");
      return herror;
    }

    // first find the bbox of _head2feet_path3D and deduce the head height
    geometry_utils::Rect3_<float> _head2feet_bbox3D =
        geometry_utils::boundingBox_vec3<float, Pt3f, std::vector<Pt3f> >(_head2feet_path3D);
    _user_height = _head2feet_bbox3D.height;
    _user_ymin = _head2feet_bbox3D.y;
    double head_height = _user_height / 8.,
        y_min = _user_ymin + head_height,
        y_max = y_min + 1.5 * head_height;

    // DEBUG: use the whole shape
    y_min = _user_ymin; y_max = y_min + _user_height;

    debugPrintf("_user_height:%g m, head_height:%g m, breast in [%g, %g]\n",
                _user_height, head_height, y_min, y_max);
    if (head_height < .1 || head_height > .45) {
      printf("convert_head2feet_path_to_feature(): "
             "head_height %g m out of bounds! "
             "(_head2feet_path3D has %i pts, _user_height:%g m)\n",
             head_height, _head2feet_path3D.size(), _user_height);
      return herror;
    }

    // project all points of _head2feet_path3D that correspond to the breast
    // in the (YZ) plane i.e. remove X
    _breast3D_Y.clear();
    _breast3D_Z.clear();
    unsigned int npts_head2feet = _head2feet_path3D.size();
    debugPrintf("_head2feet_path3D:%i pts\n", npts_head2feet);
    for (unsigned int pt_idx = 0; pt_idx < npts_head2feet; ++pt_idx) {
      double curr_y = _head2feet_path3D[pt_idx].y;
      if (curr_y <= y_min || curr_y >= y_max)
        continue;
      _breast3D_Y.push_back(curr_y);
      _breast3D_Z.push_back(_head2feet_path3D[pt_idx].z);
    } // end loop pt_idx
    if (_breast3D_Y.empty()) {
      printf("convert_head2feet_path_to_feature(): Empty _breast3D_Y!\n");
      return herror;
    }

    // for each of the bins, find the median point along the Z axis in that bin
    IndexDoubleConverter _head2feet_path_to_bins_conv(y_min, y_max, NBINS);
    unsigned int npts_breast = _head2feet_path3D.size();
    debugPrintf("npts_breast:%i\n", npts_breast);
    std::vector< std::vector<double> > binsZ;
    binsZ.resize(NBINS, std::vector<double>());
    for (unsigned int pt_idx = 0; pt_idx < npts_breast; ++pt_idx) {
      int curr_bin = _head2feet_path_to_bins_conv.double2index(_breast3D_Y[pt_idx]);
      if (curr_bin < 0 || curr_bin >= (int) NBINS)
        continue;
      debugPrintf("y:%g -> bin %i\n", _breast3D_Y[pt_idx], curr_bin);
      binsZ[curr_bin].push_back(_breast3D_Z[pt_idx]);
    } // end loop pt_idx
    // display bins
    for (unsigned int bin_idx = 0; bin_idx < NBINS; ++bin_idx) {
      debugPrintf("bin %i:'%s'\n", bin_idx, StringUtils::iterable_to_string(binsZ[bin_idx]).c_str());
    }

    // find the median point along the Z axis
    double avgZ = median(_breast3D_Z.begin(), _breast3D_Z.end());
    debugPrintf("avgZ:%g\n", avgZ);
    // these medians now make our feature
    _feature.resize(NBINS);
    for (unsigned int bin_idx = 0; bin_idx < NBINS; ++bin_idx) {
      std::vector< double >* bin_values = &(binsZ[bin_idx]);
      if (bin_values->empty())
        _feature[bin_idx] = 0;
      else
        _feature[bin_idx] = -avgZ + median(bin_values->begin(), bin_values->end());
      //_feature[bin_idx] = -avgZ + *std::max_element(bin_values->begin(), bin_values->end());
    } // end loop bin_idx
    debugPrintf("feature:'%s'\n", StringUtils::iterable_to_string(_feature).c_str());
    normalize(_feature);
    debugPrintf("feature:'%s'\n", StringUtils::iterable_to_string(_feature).c_str());

    return predict_svm(method);
  } // end convert_head2feet_path_to_feature()

  //////////////////////////////////////////////////////////////////////////////

  inline bool svm_ptr(Method method, cv::SVM* & ptr, SvmStatus* & status) {
    if (method == WALK3D) {
      ptr = &_svm_walk_3d;
      status = &_svm_walk_3d_status;
      return true;
    }
    if (method == REPROJECT) {
      ptr = &_svm_reproject;
      status = &_svm_reproject_status;
      return true;
    }
    if (method == TEMPLATE_MATCHING) {
      ptr = &_svm_template_matching;
      status = &_svm_template_matching_status;
      return true;
    }
    printf("Unknown method %i\n", method);
    ptr = NULL;
    status = NULL;
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////

  //! the success is visible accessing SVM  status
  inline bool load_svm(Method method,
                       bool display_svm = false) {
    printf("load_svm(%i)\n", method);
    cv::SVM* SVM_ptr;
    SvmStatus* status;
    if (!svm_ptr(method, SVM_ptr, status))
      return false;
#if 0 // directly load SVM
    try {
      _svm.load(SVM_FILE(TEMPLATE_MATCHING), "svm");
      _svm_is_loaded = (_svm.get_var_count() > 0);
    } catch (cv::Exception e) {
      printf("BreastDetector: could not load SVM in '%s'\n", e.what());
      _svm_is_loaded = false;
    }
#else // load training data
    cv::Mat1f training_mat, labels_mat;
    cv::FileStorage fs(SVM_FILE(method), cv::FileStorage::READ);
    if (!fs.isOpened())
      *status = SVM_STATUS_LOADED_FAILED;
    else {
      fs["labels_mat"] >> labels_mat;
      fs["training_mat"] >> training_mat;
      // status is filled inside train_svm()
      train_svm(method, training_mat, labels_mat);
      if (display_svm)
        display_svm_samples(training_mat, labels_mat);
    }
#endif
    if (*status == SVM_STATUS_LOADED_FAILED)
      printf("BreastDetector: could not load SVM in '%s'\n", SVM_FILE(method));
    return (*status == SVM_STATUS_TRAINED_SUCCESFULLY);
  } // end load_svm();

  //////////////////////////////////////////////////////////////////////////////

  bool train_svm(Method method,
                 const cv::Mat1f & training_mat,
                 const cv::Mat1f & labels_mat) {
    // train SVM
    printf("train(method:%i): training SVM from %i samples...\n",
           method, labels_mat.rows);

    cv::SVM* SVM_ptr;
    SvmStatus* status;
    if (!svm_ptr(method, SVM_ptr, status))
      return false;

    // Set up SVM's parameters
    // from http://bytefish.de/blog/machine_learning_opencv/
    // cf also http://docs.opencv.org/modules/ml/doc/support_vector_machines.html#cvsvmparams
    CvSVMParams param;
    param.svm_type    = CvSVM::C_SVC;
    //param.kernel_type = CvSVM::LINEAR;
    param.kernel_type = CvSVM::RBF;
    // 100, 1E-2: 63%
    param.C = 1; // the bigger, the bigger the penalty for misclassification
    param.gamma = 1E-1; // the smaller, the bigger the blobs
    param.term_crit.type = CV_TERMCRIT_ITER +CV_TERMCRIT_EPS;
    param.term_crit.max_iter = 1000;
    param.term_crit.epsilon = 1e-6;

    //if (!SVM_ptr->train(training_mat, labels_mat, cv::Mat(), cv::Mat(), param)) {
    if (!SVM_ptr->train_auto(training_mat, labels_mat, cv::Mat(), cv::Mat(), CvSVMParams())) {
      printf("train(method:%i): training SVM returned an error\n", method);
      *status = SVM_STATUS_LOADED_FAILED;
      return false;
    }
    printf("train(method:%i): training SVM done\n", method);
    *status = SVM_STATUS_TRAINED_SUCCESFULLY;
    return true;
  } // end train_svm()

  //////////////////////////////////////////////////////////////////////////////

  bool save_svm(Method method,
                const cv::Mat1f & training_mat,
                const cv::Mat1f & labels_mat) {
    // save to file - the SVM was trained succesfully
    cv::SVM* SVM_ptr;
    SvmStatus* status;
    if (!svm_ptr(method, SVM_ptr, status))
      return false;
    if (*status == SVM_STATUS_TRAINED_SUCCESFULLY)
      SVM_ptr->save(SVM_FILE(method), "svm");
    cv::FileStorage fs(SVM_FILE(method), cv::FileStorage::APPEND);
    if (!fs.isOpened()) {
      printf("Could not open SVM file '%s'!\n", SVM_FILE(method));
      return false;
    }
    fs << "training_mat" << training_mat;
    fs << "labels_mat" << labels_mat;
    return true;
  }

  //////////////////////////////////////////////////////////////////////////////

  //! makes use of _feature
  HeightBreast predict_svm(Method method) {
    HeightBreast herror;
    herror.gender_confidence = herror.gender = ERROR;

    cv::SVM* SVM_ptr;
    SvmStatus* status;
    if (!svm_ptr(method, SVM_ptr, status))
      return herror;
    if (*status == SVM_STATUS_NOT_LOADED) // never loaded? load it
      load_svm(method, false); // status will be check afterwards

    if (*status != SVM_STATUS_TRAINED_SUCCESFULLY) {
      printf("detect_breast_template_matching(): svm was not loaded, cannot predict gender!\n");
      return herror;
    }

    // now evaluate the best pattern _best_tvec
    cv::Mat1f query(_feature.size(), 1); // rows, cols
    for (unsigned int bin_idx = 0; bin_idx < _feature.size(); ++bin_idx)
      query.at<float>(bin_idx, 0) = _feature[bin_idx];
    //std::cout << "query:" << query << std::endl;

    float out_label = SVM_ptr->predict(query);
    // printf("out_label:%g\n", out_label);
    HeightBreast hans;
    hans.gender = (out_label < .5 ? MALE : FEMALE);
    hans.gender_confidence = 1;
    return hans;
  } // end predict_svm()

  //////////////////////////////////////////////////////////////////////////////

  HeightBreast detect_breast_walk3d(const cv::Mat1f & depth,
                                    const cv::Mat1b & user_mask,
                                    const image_geometry::PinholeCameraModel & depth_cam_model) {
    TIMER_RESET(_timer);
    HeightBreast hans;
    hans.gender_confidence = hans.gender = ERROR;

    // find height in pixels
    HeightDetector::Height h = _height_detector.height_pixels(user_mask, true); // we want confidence
    hans.height_px = h.height_px;
    hans.height_confidence = h.height_confidence;
    if (h.height_px == HeightDetector::ERROR || h.height_px == HeightDetector::NOT_COMPUTED) {
      printf("detect_breast_walk3d(): Error in height_pixels()\n");
      return hans;
    }

    // now use the dist map of _filler to find the path from _h to feet
    const cv::Mat1s & filler_queued = _height_detector.get_filler_queue();
    filler_queued.copyTo(_path_finder.get_queued());
    bool ok = _path_finder.find( filler_queued,
                                 _height_detector.get_head_pos(),
                                 _height_detector.get_foot_pos(),
                                 _head2feet_path, true, true); // propagation already done!
    if (!ok) {
      printf("detect_breast_walk3d(): Error in _path_finder.find(): no path from head fo foot!\n");
      return hans;
    }
    unsigned int head2feet_pathsize = _head2feet_path.size();

    // reproject points
    _head2feet_path3D.clear();
    for (unsigned int pt_idx = 0; pt_idx < head2feet_pathsize; ++pt_idx) {
      Pt3f pt3D = kinect_openni_utils::pixel2world_depth<Pt3f>
                  (_head2feet_path[pt_idx], depth_cam_model, depth);
      if (isnan(pt3D.x) || isnan(pt3D.y) || isnan(pt3D.z))
        continue;
      _head2feet_path3D.push_back(pt3D);
    }

    // rectify the cloud
    if (reproject_user_to_3D(depth, user_mask, depth_cam_model, false, false)
        && _tilter.learn_angles(_user_3D)) {
      _tilter.apply_learnt_angles(_head2feet_path3D);
    }
    else
      printf("detect_breast_walk3d(): Could not find the tilt for the user3D cloud!\n");

#if 0
    // now make some basic matching for the torso zone
    // man:   the body is 8 heads high, torso between 5     (navel) and 7 (chin)
    // wommn: the body is 8 heads high, torso between 4+5/6 (navel) and 7 (chin)
    double head_size_px = hans.height_px / 8;
    int min_filler_dist = hans.height_px - 1. * 6.5 * head_size_px,
        max_filler_dist = hans.height_px - 1. * (4+5/6) * head_size_px;
    int min_pt_idx = 0, max_pt_idx = head2feet_pathsize - 1;
    for (unsigned int pt_idx = 0; pt_idx < head2feet_pathsize; ++pt_idx) {
      short filler_dist = filler_queued(_head2feet_path[pt_idx]);
      if (filler_dist >= min_filler_dist && min_pt_idx == 0) // reaching chin: start adding point
        min_pt_idx = pt_idx;
      if (filler_dist >= max_filler_dist) { // gone far enough -> stop
        max_pt_idx = pt_idx;
        break;
      }
    } // end loop pt_idx
    // copy to _breast_path and _breast_path3D
    _breast_path.clear();
    _breast_path.reserve(max_pt_idx - min_pt_idx);
    for (int i = min_pt_idx; i < max_pt_idx; ++i)
      _breast_path.push_back(_head2feet_path[i]);
    _breast_path3D.clear();
    _breast_path3D.reserve(_breast_path.size());
    for (int i = min_pt_idx; i < max_pt_idx; ++i)
      _breast_path3D.push_back(_head2feet_path3D[i]);

    _breast_bbox3D =
        geometry_utils::boundingBox_vec3<float, Pt3f, std::vector<Pt3f> >(_breast_path3D);

    // project to an image on the (YZ) plan (remove X)
    int cols = 32;
    _breast_projected_2D.clear();
    _breast_projected_2D.reserve(_breast_path3D.size());
    double rowmult = (cols-1) / _breast_bbox3D.height,
        colmult = (cols-1) / _breast_bbox3D.depth,
        maxz = _breast_bbox3D.z + _breast_bbox3D.depth;
    if (KEEP_RATIO) {
      rowmult = colmult = std::min(rowmult, colmult);
      colmult *= 3; // increase breast size!
    }
    for (unsigned int pt_idx = 0; pt_idx < _breast_path3D.size(); ++pt_idx) {
      Pt2i new_pt((maxz - _breast_path3D[pt_idx].z) * colmult,
                  (_breast_path3D[pt_idx].y - _breast_bbox3D.y) * rowmult);
      // do not add the same point again and again
      if (_breast_projected_2D.size() > 0 && new_pt == _breast_projected_2D.back())
        continue;
      _breast_projected_2D.push_back(new_pt);
    } // end loop pt_idx
    _breast_projected_img.create(cols, cols);
    _breast_projected_img.setTo(0);
    image_utils::drawPolygon(_breast_projected_img, _breast_projected_2D, false,
                             cv::Scalar::all(255), 1);
    //  if (!_breast_projected_2D.empty())
    //    cv::imwrite("breast_projected_img.png", _breast_projected_img);
    bool compare_success = _breast_comparer.compareFile(_breast_projected_img);
    if (!compare_success) {
      printf("Error in _breast_comparer.compareFile()\n");
      return hans;
    }
    hans.gender = (_breast_comparer.get_best_index() == MALE ? MALE : FEMALE);
    hans.gender_confidence = _breast_comparer.get_confidence(5);
    // printf("hans.dist2breast:%g\n", _breast_comparer.getBestResult());

    //ROS_INFO_THROTTLE(1, "Time for detect: %g ms.", timer.getTimeMilliseconds());
    printf("results:%s.\n", _breast_comparer.results_to_string().c_str());
#else
    HeightBreast h2 = convert_head2feet_path_to_feature(WALK3D);
    hans.gender = h2.gender;
    hans.gender_confidence = h2.gender_confidence;
#endif
    TIMER_PRINT_RESET(_timer, "detect_breast_walk3d()");
    return hans;
  } // end detect_breast_walk3d()

  //////////////////////////////////////////////////////////////////////////////

  HeightBreast detect_breast_reproject(const cv::Mat1f & depth,
                                       const cv::Mat1b & user_mask,
                                       const image_geometry::PinholeCameraModel & depth_cam_model) {
    TIMER_RESET(_timer);
    HeightBreast hans;
    hans.gender_confidence = hans.gender = ERROR;

    // reproject
    if (!reproject_user_to_3D(depth, user_mask, depth_cam_model, false))
      return hans;
    unsigned int npts = _user_3D.size();
    if (npts == 0) {
      printf("HeightBreast: input images empty!\n");
      return hans;
    }
    Pt3f translation;
    double rotation;
    if (!fit_ellipse_and_rotate(translation, rotation))
      return hans;

    //geometry_utils::rotate_translate_polygon();
    double cos_angle = cos(rotation), sin_angle = sin(rotation);
#if 0
    _proj_rot_xmin = _proj_rot_xmax = _proj_rot_ymin = _proj_rot_ymax = 0;

    std::vector<Pt2f> edge_f;
    static const double MAX_DIFF_TO_AXIS = .05;
    for (unsigned int pt_idx = 0; pt_idx < npts; ++pt_idx) {
      double xtr = _proj_x[pt_idx] + translation.x,
          ytr = _proj_y[pt_idx] + translation.y,
          newx = ROTATE_COSSIN_X(xtr, ytr, cos_angle, sin_angle),
          newy = ROTATE_COSSIN_Y(xtr, ytr, cos_angle, sin_angle);
      if (fabs(newy) < MAX_DIFF_TO_AXIS)
        edge_f.push_back(Pt2f(newx, _user_3D[pt_idx].y));
      // refresh bounding box
      if (_proj_rot_xmin > newx)
        _proj_rot_xmin = newx;
      else if (_proj_rot_xmax < newx)
        _proj_rot_xmax = newx;
    } // end loop pt_idx

    // convert to bbox
    unsigned npts_edge = edge_f.size();
    int ncols = 32, nrows = 32;
    std::vector<int> max_col(nrows, 0);
    double proj_rot_xmean = .5*(_proj_rot_xmin+_proj_rot_xmax);
    double colmult = 1. * ncols / (_proj_rot_xmax - _proj_rot_xmin);
    double rowmult = 1. * nrows / _user_height;
    // keep ratio
    rowmult = colmult = std::min(rowmult, colmult);
    colmult *= 3; // increase breast size!
    for (unsigned int pt_idx = 0; pt_idx < npts_edge; ++pt_idx) {
      int col = (proj_rot_xmean - edge_f[pt_idx].x) * colmult + ncols / 2;
      if (col < 0 || col >= ncols)
        continue;
      int row = (_user_ymax - edge_f[pt_idx].y) * rowmult;
      if (row < 0 || row >= nrows)
        continue;
      if (!max_col[row] || max_col[row] < col)
        max_col[row] = col;
    } // end loop pt_idx

    _proj_edge.clear();
    _proj_edge.reserve(nrows);
    for (int row = 0; row < nrows; ++row) {
      if (max_col[row])
        _proj_edge.push_back(Pt2i(max_col[row], row));
    } // end loop row

    _proj_rot2img.create(nrows, ncols);
    _proj_rot2img.setTo(0);

    hans.height_px = 1;
    hans.gender = MALE;
#else
    _head2feet_path3D.clear();
    static const double MAX_DIFF_TO_AXIS = .05;
    for (unsigned int pt_idx = 0; pt_idx < npts; ++pt_idx) {
      double xtr = _proj_x[pt_idx] + translation.x,
          ytr = _proj_y[pt_idx] + translation.y,
          newx = ROTATE_COSSIN_X(xtr, ytr, cos_angle, sin_angle),
          newy = ROTATE_COSSIN_Y(xtr, ytr, cos_angle, sin_angle);
      if (fabs(newy) >= MAX_DIFF_TO_AXIS)
        continue;
      _head2feet_path3D.push_back(Pt3f(newy, _user_3D[pt_idx].y, newx));
    } // end loop pt_idx
    HeightBreast h2 = convert_head2feet_path_to_feature(REPROJECT);
    hans.gender = h2.gender;
    hans.gender_confidence = h2.gender_confidence;
#endif
    TIMER_PRINT_RESET(_timer, "detect_breast_reproject()");
    return hans;
  } // end detect_breast_reproject()

  //////////////////////////////////////////////////////////////////////////////

  /*!
  plot a given equation in gnuplot:
  y = a*( exp(-(x-s)**2) + exp(-(x+s)**2) ) + b*exp(-x**2)
  set xrange [-5:5] ; b=0; s=2; plot for [a=1:10] a*( exp(-(x-s)**2) + exp(-(x+s)**2) ) + b*exp(-x**2) title 'a='.a
  set xrange [-5:5] ; a=1; s=2; plot for [b=1:10] a*( exp(-(x-s)**2) + exp(-(x+s)**2) ) + b*exp(-x**2) title 'b='.b
  set xrange [-5:5] ; b=0; a=1; plot for [s=1:10] a*( exp(-(x-.1*s)**2) + exp(-(x+.1*s)**2) ) + b*exp(-x**2) title 's='.s
  */
  inline static double template_matching_fn(const double & x,
                                            const double & a,
                                            const double & b,
                                            const double & s) {
    // like the article
    //    double xsm = x-s, xsp = x+s;
    //    return a * (exp(-xsm*xsm) + exp(-xsp*xsp)) + b * exp(-x*x);
    // proper scale
    double xs = x*s, xsm = xs-1, xsp = xs+1;
    return a * (exp(-xsm*xsm) + exp(-xsp*xsp)) + b * exp(-xs*xs);
  }

  inline void template_matching_fn(Template & t, bool use_symmetry = true) {
    t.pts.resize(TEMPLATE_NPTS);
    for (unsigned int i = 0; i < TEMPLATE_NPTS; ++i) {
      t.pts[i].y = _template_converter.index2double(i);
      if (use_symmetry && i > TEMPLATE_NPTSHALF) // use the symmetry of the function
        t.pts[i].x = t.pts[TEMPLATE_NPTSM - i].x;
      else
        t.pts[i].x = template_matching_fn(t.pts[i].y, t.a, t.b, t.s);
    }
  } // end template_matching_fn()

  //////////////////////////////////////////////////////////////////////////////

  HeightBreast detect_breast_template_matching
  (const cv::Mat1f & depth,
   const cv::Mat1b & user_mask,
   const image_geometry::PinholeCameraModel & depth_cam_model)
  {
    static const unsigned int NSLICES = 15;
    HeightBreast hans;
    hans.gender_confidence = hans.gender = ERROR;

    TIMER_RESET(_timer);
    // reproject
    if (!reproject_user_to_3D(depth, user_mask, depth_cam_model, false, true))
      return hans;
    unsigned int npts = _user_3D.size();
    if (npts == 0) {
      printf("HeightBreast: input images empty!\n");
      return hans;
    }
    TIMER_PRINT_RESET(_timer, "reproject_user_to_3D()");
    Pt3f translation;
    double rotation;
    if (!fit_ellipse_and_rotate(translation, rotation))
      return hans;
    double cos_angle = cos(rotation), sin_angle = sin(rotation);
    TIMER_PRINT_RESET(_timer, "fit_ellipse_and_rotate()");

    // get slices on the ground (xz), so remove y
    double head_height = _user_height / 8.;
    // printf("head_size:%g\n", head_size);
    double y_min = _user_ymin + head_height, y_max = y_min + 1.5 * head_height,
        breast_width = 1.5 * head_height;
    IndexDoubleConverter slice_converter(y_min, y_max, NSLICES);
    cv::Mat1f slice_mat(NSLICES, TEMPLATE_NPTS, INFINITY); // rows, cols
    for (unsigned int pt_idx = 0; pt_idx < npts; ++pt_idx) {
      double y = _user_3D[pt_idx].y;
      if (y < y_min || y >= y_max) // not in breast vertical window
        continue;
      int slice_idx = slice_converter.double2index(y);
      if (slice_idx < 0 || slice_idx >= (int) NSLICES) {
        printf("Incorrect slice_idx %i (should be in [0, %i[\n", slice_idx, NSLICES);
        continue;
      }
      double xtr = _proj_x[pt_idx] + translation.x,
          ytr = _proj_y[pt_idx] + translation.y,
          newx = ROTATE_COSSIN_X(xtr, ytr, cos_angle, sin_angle),
          newy = ROTATE_COSSIN_Y(xtr, ytr, cos_angle, sin_angle);
      if (fabs(newy) >= breast_width) // not in breast horizontal window
        continue;
      int col = _template_converter.double2index(newy);
      if (col < 0 || col >= (int) TEMPLATE_NPTS)
        continue;
      // keep the new edge at that point - at(row,col)
      float* slice_mat_val = &slice_mat.at<float>(slice_idx, col);
      if (isinf(*slice_mat_val) || *slice_mat_val < newx)
        *slice_mat_val = newx;
    } // end loop pt_idx
    TIMER_PRINT_RESET(_timer, "make slices");

    // now translate each slice so that min value is zero
    _slices.clear();
    _slices.resize(NSLICES);
    double min_val, max_val;
    cv::minMaxLoc(slice_mat, &min_val, &max_val);
    for (unsigned int slice_idx = 0; slice_idx < NSLICES; ++slice_idx) {
      std::vector<Pt2f>* slice = &(_slices[slice_idx]);
      //cv::minMaxLoc(slice_mat.row(slice_idx), &min_val, &max_val);
      float* slice_mat_data = (float*) slice_mat.ptr(slice_idx);
      for (unsigned int col = 0; col < TEMPLATE_NPTS; ++col) {
        if (!isinf(slice_mat_data[col])) // max was set before
          slice->push_back(Pt2f(slice_mat_data[col] - min_val, // x (depth)
                                _template_converter.index2double(col))); // y
      }
    } // end loop slice_idx
    TIMER_PRINT_RESET(_timer, "translate slices");

    if (!detect_breast_template_matching_compare_slices())
      return hans;
    TIMER_PRINT_RESET(_timer, "detect_breast_template_matching_compare_slices()");

    // store our feature
    _feature.resize(3);
    _feature[0] = _best_template->a;
    _feature[1] = _best_template->b;
    _feature[2] = _best_template->s;

    return predict_svm(TEMPLATE_MATCHING);
  } // end detect_breast_template_matching()

  //////////////////////////////////////////////////////////////////////////////

  inline void detect_breast_template_matching_precompute_templates() {
    printf("detect_breast_template_matching_precompute_templates()\n");
    static const double MIN_A = 0, MAX_A = .5, A_INCR = .02, // .05
        MIN_B = -.5, MAX_B = .5, B_INCR = .05, // .1
        MIN_S = 0, MAX_S = 10, S_INCR = .5; // 1
    int ntemplates = (MAX_A - MIN_A) / A_INCR +  (MAX_B - MIN_B) / B_INCR
                     + (MAX_S - MIN_S) / S_INCR;
    _precomputed_templates.clear();
    _precomputed_templates.reserve(ntemplates);
    for (double a = MIN_A; a <= MAX_A; a+=A_INCR) {
      for (double b = MIN_B; b <= MAX_B; b+=B_INCR) {
        for (double s = MIN_S; s <= MAX_S; s+=S_INCR) {
          BreastDetector::Template t(a, b, s);
          BreastDetector::template_matching_fn(t);
          _precomputed_templates.push_back(t);
        } // end loop s
      } // end loop b
    } // end loop a
  } // end detect_breast_template_matching_precompute_templates();

  //////////////////////////////////////////////////////////////////////////////

  inline bool detect_breast_template_matching_compare_slices() {
#if 0 // precomputed templates
    if (_precomputed_templates.empty()) // load templates the first time
      detect_breast_template_matching_precompute_templates();

    unsigned int ntemplates = _precomputed_templates.size(), nslices = _slices.size();
    double best_dist = INFINITY;
    for (unsigned int template_idx = 0; template_idx < ntemplates; ++template_idx) {
      BreastDetector::Template* t = &(_precomputed_templates[template_idx]);
      for (unsigned int slice_idx = 0; slice_idx < nslices; ++slice_idx) {
        // cv::matchShapes() returns a positive value
        //            double curr_dist = -cv::matchShapes(tvec, _slices[slice_idx],
        //                                               CV_CONTOURS_MATCH_I2, 0);
        double curr_dist =
            hausdorff_distances::D22_with_min<Pt2f, std::vector<Pt2f> >
            (t->pts, _slices[slice_idx], best_dist, hausdorff_distances::dist_L1_double);
        //printf("curr_dist:%g\n", curr_dist);
        if (curr_dist < best_dist) {
          best_dist = curr_dist;
          _best_slice_idx = slice_idx;
          _best_template = t;
          // printf("New best! best_dist:%g, a:%g, b:%g, s:%g\n", best_dist, a, b, s);
          //      printf("tvec:'%s', _slices[slice_idx]:'%s'\n",
          //             StringUtils::iterable_to_string(tvec).c_str(),
          //             StringUtils::iterable_to_string(_slices[slice_idx]).c_str());
        }
      } // end loop slice_idx
    } // end loop template_idx


    return (best_dist != INFINITY);
#else //use octave
    if (_slices.empty()) {
      printf("_slices.empty()!\n");
      return false;
    }
    // prepair the octave command
    unsigned int nslices = _slices.size();
    std::ostringstream instr, X_str, Y_str;
    // add all x
    for (unsigned int slice_idx = 0; slice_idx < nslices; ++slice_idx) {
      unsigned int npts = _slices[slice_idx].size();
      X_str << "["; Y_str << "[";
      for (unsigned int pt_idx = 0; pt_idx < npts; ++pt_idx) {
        X_str << _slices[slice_idx][pt_idx].y << (pt_idx < npts - 1 ? ", " : "");
        Y_str << _slices[slice_idx][pt_idx].x << (pt_idx < npts - 1 ? ", " : "");
      } // end loop pt_idx
      X_str << "]"; Y_str << "]";
    } // end loop slice_idx

    instr << "octave --silent --norc --eval \"tic; global Xarray = {"
          << X_str.str() << "}; global Yarray = {"
          << Y_str.str() << "}; source "
          << ros::package::getPath("people_recognition_vision")
          << "/src/height_detector/template_regression.m; toc\"";

    // call octave (system call)
    //printf("instr:'%s'\n", instr.str().c_str());
    std::string octave_output = system_utils::exec_system_get_output(instr.str().c_str());
    while (StringUtils::find_and_replace(octave_output, " =", "=")) {}
    while (StringUtils::find_and_replace(octave_output, "\n", " ")) {}
    while (StringUtils::find_and_replace(octave_output, "  ", " ")) {}
    //printf("octave_output:'%s'\n", octave_output.c_str());

    // default values
    _best_slice_idx = 0;
    _precomputed_templates.resize(1, Template(0, 0, 0));
    _best_template = &(_precomputed_templates.front());
    // parse results
    std::vector<std::string> words;
    StringUtils::StringSplit(octave_output, " ", &words);
    int nwords = words.size(), success_idx = nwords-1;
    while (success_idx >= 0 && words[success_idx] != "success=")
      --success_idx;
    if (success_idx < 0
        || success_idx + 13 > nwords-1 // we read 13 values
        || StringUtils::cast_from_string<int>(words[success_idx+1]) <= 0) {
      printf("Octave failed in optimizing!\n");
      return false;
    }
    _best_template->a = StringUtils::cast_from_string<double>(words[success_idx+3]);
    _best_template->b = StringUtils::cast_from_string<double>(words[success_idx+4]);
    _best_template->s = StringUtils::cast_from_string<double>(words[success_idx+5]);
    _best_slice_idx = StringUtils::cast_from_string<int>(words[success_idx+7]);
    double best_error = StringUtils::cast_from_string<double>(words[success_idx+9]);
    double time = 1000 * StringUtils::cast_from_string<double>(words[success_idx+13]);
    BreastDetector::template_matching_fn(*_best_template, false);
    printf("after octave: time:%g s, a:%g, b:%g, c:%g, best_error:%g\n",
           time, _best_template->a, _best_template->b, _best_template->s, best_error);
    return true;
#endif
  } // end detect_breast_template_matching_compare_slices()

  //////////////////////////////////////////////////////////////////////////////

  bool breast2img_walk3d(const cv::Mat1b & user_mask,
                         cv::Mat3b & breast_illus,
                         HeightBreast h) {
    // draw skeleton
    _height_detector.height2img(user_mask, breast_illus, true, h);
    // draw _head2feet_path
    image_utils::drawListOfPoints(breast_illus, _head2feet_path, cv::Vec3b(255, 255, 255));
    // draw _breast_path
    cv::Rect breast_path_bbox =
        geometry_utils::boundingBox_vec<std::vector<Pt2i>, cv::Rect>(_breast_path);
    cv::rectangle(breast_illus, breast_path_bbox, CV_RGB(0, 0, 255), 2);

    // print estimated gender and dist
    cv::Point txt_pos = _height_detector.get_head_pos() + cv::Point(0, -45);
    std::ostringstream txt;
    cv::Scalar color = cv::Scalar::all(255);
    if (h.gender == MALE) {
      txt << "M";
      color = CV_RGB(100, 100, 255); // blue
    }
    else if (h.gender == FEMALE) {
      txt << "F";
      color = CV_RGB(255, 100, 100); // pink
    }
    else
      txt << "UNKNOWN";
    txt << " " << (int) (100 *  h.gender_confidence) << "%sure";
    cv::putText(breast_illus, txt.str(), txt_pos, CV_FONT_HERSHEY_PLAIN, 2, color, 2);

    // resize _breast_projected_img
#if 0
    if  (!_breast_projected_img.empty())
      cv::resize(_breast_projected_img, _breast_projected_big,
                 cv::Size(), 10, 10, cv::INTER_NEAREST);
    else
      _breast_projected_big.create(1, 1);
    cv::imshow("_breast_projected_big", _breast_projected_big);
#else
    //    ms.clear();
    //    ms.set_origin(cv::Point2f(0, 1.5));
    //    ms.set_scale(1. / 150);
    //    ms.draw_grid(1.f, 150);
    //    ms.draw_axes();
    //    mini_stage_plugins::plot_xy_pts(ms, _user_3D, CV_RGB(0, 255, 0), 1);
    //    mini_stage_plugins::plot_xy_pts(ms, _breast3D_Z, _breast3D_Y, CV_RGB(255, 0, 0), 2, 8, -3);
    //    cv::imshow("viz", ms.get_viz()); cv::waitKey(5);

    // draw features
    int cols = 200;
    if (!_feature.empty() && breast_illus.cols > cols && breast_illus.rows > cols) {
      cv::Rect bbox(0, 0, cols, cols);
      cv::Mat3b feature_illus = breast_illus(bbox);
      feature_illus.setTo(0);
      std::vector<cv::Point> pts;
      pts.push_back(cv::Point(cols, 0));
      // the first feature correspond to the head (low y in the illus), last to the feet
      int nfeatures = _feature.size(), height1 = cols / nfeatures;
      for (int bin_idx = 0; bin_idx < nfeatures; ++bin_idx)
        pts.push_back(cv::Point(cols/2 + _feature[bin_idx] * 30, bin_idx * height1));
      pts.push_back(cv::Point(cols, cols));
      cv::fillPoly(feature_illus, std::vector< std::vector<cv::Point> > (1, pts),
                   CV_RGB(0, 255, 255));
      cv::rectangle(breast_illus, bbox, CV_RGB(255, 255, 255), 1);
    }

#endif
    return true;
  } // end breast2img_walk3d()

  //////////////////////////////////////////////////////////////////////////////

  bool breast2img_reproject(const cv::Mat1b & user_mask,
                            cv::Mat3b & breast_illus,
                            HeightBreast h) {
    //    std::string _window_name="foo";
    //    cv::namedWindow(_window_name);
    //    ms.set_mouse_move_callback(_window_name);
    //    while (true) {
    //      printf("loop\n");
    ms.clear();
    ms.set_origin(cv::Point2f(0, 1.5));
    ms.set_scale(1. / 150);
    ms.draw_grid(1.f, 150);
    ms.draw_axes();
    mini_stage_plugins::plot_xy_pts(ms, _proj_x, _proj_y, CV_RGB(255, 100, 100), 2);
    mini_stage_plugins::plot_xy_pts(ms, _proj_rot_x, _proj_rot_y, CV_RGB(100, 255, 100), 2);
    //    cv::line(ms.get_viz(), ms.world2pixel(_ellipse_center),
    //             ms.world2pixel(_ellipse_end1), CV_RGB(0, 0, 0), 2);
    //    cv::line(ms.get_viz(), ms.world2pixel(_ellipse_center),
    //             ms.world2pixel(_ellipse_end2), CV_RGB(0, 0, 0), 2);

    // draw _ellipse
    mini_stage_plugins::draw_ellipse(ms, _ellipse, CV_RGB(0, 0, 0), 2);
    // draw _ellipse_closest_end
    cv::line(ms.get_viz(), ms.world2pixel(_ellipse_center),
             ms.world2pixel(_ellipse_closest_end), CV_RGB(0, 0, 0), 2);

    // draw bbox
    cv::rectangle(ms.get_viz(), ms.world2pixel(_proj_rot_xmin, _proj_rot_ymin),
                  ms.world2pixel(_proj_rot_xmax, _proj_rot_ymax), CV_RGB(0, 0, 0), 2);

    // resize _proj_rot2img
    double scale_factor = 10;
    if  (!_proj_rot2img.empty()) {
      cv::cvtColor(_proj_rot2img, _proj_rot2img_big, CV_GRAY2BGR);
      cv::resize(_proj_rot2img_big, _proj_rot2img_big,
                 cv::Size(), scale_factor, scale_factor, cv::INTER_NEAREST);
    } else
      _proj_rot2img_big.create(1, 1);
    // draw _proj_edge
    std::vector<Pt2i> _proj_edge_big;
    for (unsigned int pt_idx = 0; pt_idx < _proj_edge.size(); ++pt_idx)
      _proj_edge_big.push_back(scale_factor * _proj_edge[pt_idx]);
    image_utils::drawPolygon(_proj_rot2img_big, _proj_edge_big, false,
                             CV_RGB(255, 0, 0), 1);

    if (!_proj_rot2img_big.empty())
      cv::imshow("_proj_rot2img_big", _proj_rot2img_big);
    //cv::imshow("_proj_rot2img_thres", _proj_rot2img_thres);

    //      cv::imshow(_window_name, ms.get_viz());
    //      char c = cv::waitKey(50);
    //      if ((int) c == 27)
    //        break;
    //    } // end while (true)
    ms.get_viz().copyTo(breast_illus);
    return true;
  }

  //////////////////////////////////////////////////////////////////////////////

  bool breast2img_template_matching(const cv::Mat1b & user_mask,
                                    cv::Mat3b & breast_illus,
                                    HeightBreast h) {
    unsigned int nslices = _slices.size();
    if (_best_slice_idx >= nslices) {
      printf("_best_slice_idx %i out of bound!\n", _best_slice_idx);
      return false;
    }
    std::string win_name = "breast2img_template_matching";
    ms.set_scale(1. / 500);
#if 0 // display all slices in MiniStage
    ms.set_mouse_move_callback(win_name);
    for (unsigned int slice_idx = 0; slice_idx < nslices; ++slice_idx) {
      std::vector<Pt2f>* slice = &(_slices[slice_idx]);
      while((char) cv::waitKey(10) != 'q') {
        ms.clear();
        ms.draw_grid(1.f, 150);
        ms.draw_axes();
        mini_stage_plugins::plot_xy_pts(ms, _proj_x, _proj_y, CV_RGB(255, 150, 150), 2);
        mini_stage_plugins::draw_ellipse(ms, _ellipse, CV_RGB(0, 0, 0), 2);
        mini_stage_plugins::plot_xy_pts(ms, (*slice), CV_RGB(255, 0, 0), 2);
        cv::imshow(win_name, ms.get_viz());
      } // end while key
    } // end loop slice_idx
#endif

    // display the best slice in MiniStage
    unsigned int slice_idx = _best_slice_idx;
    std::vector<Pt2f>* slice = &(_slices[slice_idx]);
    unsigned int slice_npts = slice->size();
    if (slice_npts == 0) {
      printf("Best slice %i empty!\n", slice_idx);
      return false;
      //          printf("Not showing empty slice:%i\n", slice_idx);
      //          continue;
    }

    //while((char) cv::waitKey(10) != 'q') {
    ms.clear();
    ms.draw_grid(1.f, 150);
    ms.draw_axes();
    // show the initial user:ellipse and set of points
    // unsigned int user_npts = _user_3D.size();
    mini_stage_plugins::plot_xy_pts(ms, _proj_x, _proj_y, CV_RGB(255, 150, 150), 2);
    mini_stage_plugins::draw_ellipse(ms, _ellipse, CV_RGB(0, 0, 0), 2);
    //    for (unsigned int pt_idx = 0; pt_idx < user_npts; ++pt_idx)
    //      cv::circle(ms.get_viz(), ms.world2pixel(_proj_x[pt_idx], _proj_y[pt_idx]),
    //                 2, CV_RGB(255, 150, 150), -1);

    // plot the best slice and its template vec
    mini_stage_plugins::plot_xy_pts(ms, (*slice), CV_RGB(255, 0, 0), 2);
    mini_stage_plugins::plot_xy_pts(ms, _best_template->pts, CV_RGB(0,0,0), 2);
    // cv::imshow(win_name, ms.get_viz());
    //} // end while key

    ms.get_viz().copyTo(breast_illus);
    return true;
  }

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // shared data
  std::vector<Pt3f> _user_3D;
  CloudTilter _tilter;
  cv::Mat1b user_mask_eroded;
  TIMER_CREATE(_timer);

  std::vector<double> _proj_x, _proj_y;
  Pt2f _ellipse_center, _ellipse_end1, _ellipse_end2;
  Pt2f _ellipse_closest_end;
  ellipse_utils::Ellipse _ellipse;
  double _user_ymin, _user_ymax, _user_height;

  std::vector<double> _breast3D_Y, _breast3D_Z;
  static const unsigned int NBINS = 15;

  //! should contain the numerical values that are learnable or for prediction
  std::vector<double> _feature;

  // WALK3D ////////////////////////
  HeightDetector _height_detector;
  geometry_utils::Rect3_<float> _breast_bbox3D;
  image_utils::ShortestPathFinder<short> _path_finder;
  std::vector<Pt2i> _head2feet_path;
  std::vector<Pt3f> _head2feet_path3D;

  // breast
  std::vector<Pt2i> _breast_path;
  std::vector<Pt3f> _breast_path3D;
  std::vector<Pt2i> _breast_projected_2D;
  cv::Mat1b _breast_projected_img;
  ImageComparer _breast_comparer;

  cv::SVM _svm_walk_3d;
  SvmStatus _svm_walk_3d_status;

  // viz
  std::vector<cv::Vec3b> _head2feet_path_colors;
  std::vector<cv::Vec3b> _breast_path_colors;
  cv::Mat1b _breast_projected_big;

  // REPROJECT ///////////////////////
  std::vector<double> _proj_rot_x, _proj_rot_y;
  double _proj_rot_xmin, _proj_rot_xmax, _proj_rot_ymin, _proj_rot_ymax;
  cv::Mat1b _proj_rot2img;
  cv::Mat1b _proj_rot2img_thres;
  std::vector<Pt2i> _proj_edge;

  cv::SVM _svm_reproject;
  SvmStatus _svm_reproject_status;

  // viz
  MiniStage ms;
  cv::Mat3b _proj_rot2img_big;

  // TEMPLATE_MATCHING ///////////////////////
  std::vector<Pt3f> _user_3D_template_matching;
  std::vector<std::vector<Pt2f> > _slices;
  static const double TEMPLATE_xmin = -.25, TEMPLATE_xmax = .25;
  //! NPTS = 1 + (xmax - xmin) / dx
  static const unsigned int TEMPLATE_NPTS = 25,
  TEMPLATE_NPTSM = TEMPLATE_NPTS-1, TEMPLATE_NPTSHALF = TEMPLATE_NPTS/2;
  IndexDoubleConverter _template_converter;
  unsigned int _best_slice_idx;
  std::vector<Template> _precomputed_templates;
  Template* _best_template;

  cv::SVM _svm_template_matching;
  SvmStatus _svm_template_matching_status;
}; // end class BreastDetector

#endif // BREAST_DETECTOR_H
