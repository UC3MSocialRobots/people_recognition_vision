/*!
  \file        person_histogram.h
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

A structured representation of the colors in a person.
 */

#ifndef PERSON_HISTOGRAM_H
#define PERSON_HISTOGRAM_H

// AD
#include <cmatrix/cmatrix.h>
#include "kinect/kinect_openni_utils.h"
#include <visu_utils/histogram_utils.h>
#include <image_utils/content_processing.h>
#include <image_utils/drawing_utils.h>
#include <point_clouds/blob_segmenter.h>

#define SVM_USE_STD_DEV_HIST

#ifndef DEBUG_PRINT
#define DEBUG_PRINT(...)   {}
//#define DEBUG_PRINT(...)   ROS_WARN(__VA_ARGS__)
//#define DEBUG_PRINT(...)   printf(__VA_ARGS__)
#endif // DEBUG_PRINT



// the background color for the illustration images
#define PH_ILLUS_BG_COLOR_R 255
#define PH_ILLUS_BG_COLOR_G 255
#define PH_ILLUS_BG_COLOR_B 255

////////////////////////////////////////////////////////////////////////////////

class PersonHistogram {
public:
  typedef std::vector<histogram_utils::Histogram> HistVec;
  //! the number of bins for histograms
  static const int HIST_NBINS = 15;
  /*! the number of parts in the body: head, torso, legs.
   *  This will also be the number of histograms. */
  static const unsigned int BODY_PARTS = 3;
  static const int MAT_COLS =
    #ifdef SVM_USE_STD_DEV_HIST
      2 * BODY_PARTS;
#else
      HIST_NBINS;
#endif


  //////////////////////////////////////////////////////////////////////////////

  /*!
   * The lookup function for converting distance in pixels from the head top
   * to a body limb index.
   * \param propagation_pixel_dist
   * \param pixel2meters_factor
   *    in meters/pixels
   * \return
   *    1= head, 2=torso, 3=legs
   */
  static inline float body_areas_lookup_function
  (const int propagation_pixel_dist, const int, const int, void* pixel2meters_factor)
  {
    double dist_meters = *((double*) pixel2meters_factor) * propagation_pixel_dist;
    if (dist_meters < 0.15) // head
      return 1;
    else if (dist_meters < 0.3) // neck discarded
      return 0;
    else if (dist_meters < 0.6) // torso
      return 2;
    else if (dist_meters < 1.0) // waist discarded
      return 0;
    else if (dist_meters < 1.9) // legs
      return 3;
    else // more than 1.90m high? not likely -> discard
      return 0;
  } // end body_areas_lookup_function()

  //////////////////////////////////////////////////////////////////////////////

  //! empty ctor
  PersonHistogram() {
    clear();
  }

  //! empty ctor
  PersonHistogram(const PersonHistogram & in) {
    // copy all fields
    _hist_vector.resize(BODY_PARTS); // copy histogram
    for (unsigned int hist_idx = 0; hist_idx < BODY_PARTS; ++hist_idx)
      in._hist_vector[hist_idx].copyTo(_hist_vector[hist_idx]);
    in._illus_color_img.copyTo(_illus_color_img);
    _input_images_nb = in._input_images_nb;
    in._multimask.copyTo(_multimask);
    in._user_mask_bbox.copyTo(_user_mask_bbox);
    // no not copy _user_mask
    in._rgb_hsv.copyTo(_rgb_hsv);
    in._hue.copyTo(_hue);
    _seed = in._seed;
    _top_centered_seed = in._top_centered_seed;
    _user_mask_bbox_rect = in._user_mask_bbox_rect;
    in._seen_buffer_short.copyTo(_seen_buffer_short);
    in._lookup_result.copyTo(_lookup_result);
  }

  //////////////////////////////////////////////////////////////////////////////

  /*! Templated constructor for 3 parameters calling the function factury.
   * \see function factory create() for all possibilites of arguments */
  template<class _T1, class _T2, class _T3>
  PersonHistogram(const _T1 & arg1, const _T2 & arg2, const _T3 & arg3) {
    // call factory - refresh image
    create(arg1, arg2, arg3);
  }

  //////////////////////////////////////////////////////////////////////////////

  /*! Templated constructor for 4 parameters calling the function factury.
   * \see function factory create() for all possibilites of arguments */
  template<class _T1, class _T2, class _T3, class _T4>
  PersonHistogram(const _T1 & arg1, const _T2 & arg2, const _T3 & arg3, const _T4 & arg4) {
    // call factory - refresh images
    create(arg1, arg2, arg3, arg4);
  }

  ////////////////////////////////////////////////////////////////////////////////

  /*!
   * The "minimal" factory.
   * Well suited for the user mask and the pixel2meters_factor
   * were obtained with another method.
   * \param rgb
   *    image coming from the kinect
   * \param user_mask
   *    pixels where the user is in \a rgb are pixels with values are != 0
   * \param pixel2meters_factor
   *    conversion factor, in meters/pixels
   * \param want_refresh_illus_images
   *    true for copying the binary usermask in _illus_mask_img
   *    and the rgb usermask in _illus_color_img and
   * \return
   *    true if success
   */
  inline bool create (const cv::Mat3b & rgb,
                      const cv::Mat1b & user_mask,
                      const double & pixel2meters_factor,
                      bool want_refresh_illus_images = true) {
    DEBUG_PRINT("PersonHistogram::create(line %i, pixel2meters_factor:%g)\n",
                __LINE__, pixel2meters_factor);
    clear();
    TIMER_RESET(_depth_canny.timer);
    // get Hue channel
    _user_mask_bbox_rect = image_utils::boundingBox(user_mask);
    if (_user_mask_bbox_rect.x < 0)
      return false;
    // we need a clone as propagative_floodfill() needs a continuous image
    _user_mask_bbox = user_mask(_user_mask_bbox_rect).clone();
    color_utils::rgb2hue(rgb(_user_mask_bbox_rect), _rgb_hsv, _hue);
    TIMER_PRINT_RESET(_depth_canny.timer, "vec_of_hists(): rgb2hue()");

    // find top point
    _top_centered_seed = image_utils::find_top_point_centered(_user_mask_bbox, .4);
    if (_top_centered_seed.x < 0)
      return false;
    TIMER_PRINT_RESET(_depth_canny.timer, "vec_of_hists(): find_top_point_centered()");

    // propagation
    double pixel2meters_factor_clone = pixel2meters_factor;
    image_utils::propagative_floodfill
        (_user_mask_bbox, _top_centered_seed, _seen_buffer_short,
         false, // no need to search top point as it is already the one we have
         &PersonHistogram::body_areas_lookup_function,
         &_lookup_result, &pixel2meters_factor_clone);
    TIMER_PRINT_RESET(_depth_canny.timer, "vec_of_hists(): propagative_floodfill()");
    _lookup_result.convertTo(_multimask, CV_8U);
    TIMER_PRINT_RESET(_depth_canny.timer, "vec_of_hists(): conv lookup_result->multimask");


    // compute histogram - just keep a ROI
    int max_value = 180; // hue till 180
    histogram_utils::get_vector_of_histograms
        (_hue, _hist_vector, HIST_NBINS, max_value, _multimask, BODY_PARTS);
    TIMER_PRINT_RESET(_depth_canny.timer, "vec_of_hists(): get_vector_of_histograms()");
    _input_images_nb = 1;

    // keep illus images
    if (want_refresh_illus_images) {
      DEBUG_PRINT("PersonHistogram::create(line %i):  Refreshing images...\n", __LINE__);
      // _user_mask_bbox_img_clone.copyTo(_illus_mask_img);
      // fill _illus_color_img with bg color
      _illus_color_img.create(_user_mask_bbox_rect.height, _user_mask_bbox_rect.width);
      cv::Vec3b bg_color(PH_ILLUS_BG_COLOR_B, PH_ILLUS_BG_COLOR_G, PH_ILLUS_BG_COLOR_R);
      _illus_color_img.setTo(bg_color);
      // real copy
      if (!geometry_utils::bboxes_included(image_utils::bbox_full(rgb), _user_mask_bbox_rect)) {
        printf("create(): Weird bug: _user_mask_bbox_rect (%s) "
               "not included in rgb (%s)...\n",
               geometry_utils::print_rect(_user_mask_bbox_rect).c_str(),
               image_utils::infosImage(rgb).c_str());
      }
      else
        rgb(_user_mask_bbox_rect).copyTo(_illus_color_img, _user_mask_bbox);
      refresh_illus_images_frame_counter();
      TIMER_PRINT_RESET(_depth_canny.timer, "vec_of_hists(): want_refresh_illus_images");
    } // end  if want_refresh_illus_images
    return true;
  }

  ////////////////////////////////////////////////////////////////////////////////

  /*!
   * factory for when the user mask is already provided,
   * but the pixel2meters_factor is unknown.
   * Aimed at a use with NITE api, which already provides the user mask.
   * \param rgb
   *    the color image from the kinect
   * \param depth
   *    used only to determine pixel2meters_factor
   * \param user_mask
   *    the user mask as computed from outside, for instance with NITE api.
   * \param pixel2meters_arg
   *    Needed for the computation of pixel2meters_factor
   *    Can be std::string kinect_serial_number
   *    or image_geometry::PinholeCameraModel depth_camera_model
   * \param want_refresh_illus_images
   *    true for copying the binary usermask in _illus_mask_img
   *    and the rgb usermask in _illus_color_img and
   * \return
   *    true if success
   */
  template<class T>
  inline bool create(const cv::Mat3b & rgb,
                     const cv::Mat1b & user_mask,
                     const cv::Mat1f & depth,
                     const T & pixel2meters_arg,
                     bool want_refresh_illus_images = true) {
    DEBUG_PRINT("PersonHistogram::create(line %i)\n", __LINE__);
    clear();
    TIMER_RESET(_depth_canny.timer);
    // find a random seed in user_mask
#if 1
    _user_mask_bbox_rect = image_utils::boundingBox(user_mask);
    if (_user_mask_bbox_rect.x < 0) {
      printf("PersonHistogram::create(): error in boundingBox()!\n");
      return false;
    }
    _seed = geometry_utils::rect_center<cv::Rect, cv::Point>(_user_mask_bbox_rect);
    unsigned int ntries = 0, MAX_TRIES = user_mask.cols * user_mask.rows;
    while(user_mask.at<uchar>(_seed) == 0 && ntries++ < MAX_TRIES) {
      _seed.x = _user_mask_bbox_rect.x + rand() % _user_mask_bbox_rect.width;
      _seed.y = _user_mask_bbox_rect.y + rand() % _user_mask_bbox_rect.height;
    }
    if (ntries >= MAX_TRIES) {
      printf("PersonHistogram::create(): could not find a seed in user mask!\n");
      return false;
    }
    TIMER_PRINT_RESET(_depth_canny.timer, "finding a seed in user_mask");

    // get pixel2meters_factor from depth_camera_model
    double pixel2meters_factor = kinect_openni_utils::compute_pixel2meters_factor
                                 (depth, pixel2meters_arg, _seed);
    if (isnan(pixel2meters_factor)) {
      printf("PersonHistogram::create(): pixel2meters_factor at seed (%i, %i) "
             "is NaN!\n", _seed.x, _seed.y);
      return false;
    }
#else
    double pixel2meters_factor = kinect_openni_utils::compute_average_pixel2meters_factor
                                 (depth, pixel2meters_arg, user_mask);
#endif
    TIMER_PRINT_RESET(_depth_canny.timer, "pixel2meters_factor()");

    return create(rgb, user_mask, pixel2meters_factor, want_refresh_illus_images);
  }

  //////////////////////////////////////////////////////////////////////////////

  /*!
   * factory for loading rgb, depth and user mask from a given filename prefix
   * \param rgb_depth_user_filename_prefix
   * \param user_idx
   *    the idx of the user in the user mask
   * \param pixel2meters_arg
   *    Needed for the computation of pixel2meters_factor
   *    Can be std::string kinect_serial_number
   *    or image_geometry::PinholeCameraModel depth_camera_model
   * \param want_refresh_illus_images
   *    true for copying the binary usermask in _illus_mask_img
   *    and the rgb usermask in _illus_color_img
   * \return
   *    true if success
   */
  template<class T>
  bool create (const std::string & rgb_depth_user_filename_prefix,
               const uchar user_idx,
               const T & pixel2meters_arg,
               bool want_refresh_illus_images = true) {
    DEBUG_PRINT("PersonHistogram::create(line %i)\n", __LINE__);
    cv::Mat1b user_mask;
    cv::Mat rgb, depth;
    bool ok = image_utils::read_rgb_depth_user_image_from_image_file
              (rgb_depth_user_filename_prefix, &rgb, &depth, &user_mask);
    if (!ok)
      return false;
    return create<T>(rgb, (user_mask == user_idx), depth, pixel2meters_arg,
                     want_refresh_illus_images);
  }

  //////////////////////////////////////////////////////////////////////////////

  /*!
   * Factory for when we have rgb and depth,
   * but we only know one point of the user (the seed).
   * The user mask is computed thanks to compute_user_mask().
   * \param rgb
   * \param depth
   * \param seed
   * \param pixel2meters_factor
   * \param want_refresh_illus_images
   * \return
   */
  inline bool create(const cv::Mat3b & rgb,
                     const cv::Mat1f & depth,
                     const cv::Point & seed,
                     const double pixel2meters_factor,
                     bool want_refresh_illus_images = true) {
    DEBUG_PRINT("PersonHistogram::create(line %i)\n", __LINE__);
    bool ok = compute_user_mask(depth, seed);
    if (!ok)
      return false;
    return create(rgb, _user_mask, pixel2meters_factor, want_refresh_illus_images);
  }

  //////////////////////////////////////////////////////////////////////////////

  /*!
   * factory from a pair of images (rgb, depth) and a seed.
   * The user mask is computed thanks to compute_user_mask().
   * The pixel2meters_factor is computed thanks to
   *  kinect_openni_utils::compute_pixel2meters_factor().
   * \param rgb, depth
   *    images coming from the kinect
   * \param seed
   *    a pixel (x, y) in rgb where the user is supposed to be.
   *    Used for the user mask computation (floodfilling fro this seed in edge image).
   * \param pixel2meters_arg
   *    Needed for the computation of pixel2meters_factor
   *    Can be std::string kinect_serial_number
   *    or image_geometry::PinholeCameraModel depth_camera_model
   * \param want_refresh_illus_images
   *    true for copying the binary usermask in _illus_mask_img
   *    and the rgb usermask in _illus_color_img and
   * \return
   *    true if success
   */
  template<class T>
  inline bool create (const cv::Mat3b & rgb,
                      const cv::Mat1f & depth,
                      const cv::Point & seed,
                      const T & pixel2meters_arg,
                      bool want_refresh_illus_images = true) {
    DEBUG_PRINT("PersonHistogram::create(line %i)\n", __LINE__);
    clear();
    TIMER_RESET(_depth_canny.timer);
    double pixel2meters_factor =
        kinect_openni_utils::compute_pixel2meters_factor(depth, pixel2meters_arg, seed);
    if (isnan(pixel2meters_factor)) {
      printf("PersonHistogram::create(): pixel2meters_factor at seed (%i, %i) "
             "is NaN!\n", seed.x, seed.y);
      return false;
    }
    TIMER_PRINT_RESET(_depth_canny.timer, "pixel2meters_factor()");
    return create(rgb, depth, seed, pixel2meters_factor, want_refresh_illus_images);
  }

  //////////////////////////////////////////////////////////////////////////////

  /*!
   * factory from data stored in files and a seed
   * \param rgb_depth_filename_prefix
   *    where to find saved files of images coming from the kinect
   * \param seed
   *    a pixel (x, y) in color where the user is supposed to be.
   *    Used for the user mask computation (floodfilling fro this seed in edge image).
   * \param pixel2meters_arg
   *    Needed for the computation of pixel2meters_factor
   *    Can be std::string kinect_serial_number
   *    or image_geometry::PinholeCameraModel depth_camera_model
   * \param want_refresh_illus_images
   *    true for copying the binary usermask in _illus_mask_img
   *    and the rgb usermask in _illus_color_img and
   * \return
   *    true if success
   */
  template<class T>
  bool create (const std::string & rgb_depth_filename_prefix,
               const cv::Point & seed,
               const T & pixel2meters_arg,
               bool want_refresh_illus_images = true) {
    DEBUG_PRINT("PersonHistogram::create(line %i)\n", __LINE__);
    clear();
    cv::Mat rgb, depth;
    if (!image_utils::read_rgb_and_depth_image_from_image_file(rgb_depth_filename_prefix, &rgb, &depth))
      return false;
    return create<T>(rgb, depth, seed, pixel2meters_arg, want_refresh_illus_images);
  }

  //////////////////////////////////////////////////////////////////////////////

  /*! Templated constructor for 3 vectors parameters.
   *  Making all histograms separately then merging them.
   * \see function factory create() for all possibilites of arguments */
  template<class _T1, class _T2, class _T3>
  bool create(const std::vector<_T1> & v1,
               const std::vector<_T2> & v2,
               const std::vector<_T3> & v3,
               bool want_refresh_illus_images = true)
  {
    DEBUG_PRINT("PersonHistogram::create(line %i)\n", __LINE__);
    clear();
    unsigned int nhists = v1.size();
    std::vector<PersonHistogram> hists(nhists, PersonHistogram());
    for (unsigned int hist_idx = 0; hist_idx < nhists; ++hist_idx) {
      // only refresh for last image (used for merge)
      bool refres_images = want_refresh_illus_images && (hist_idx == nhists - 1);
      hists[hist_idx].create(v1[hist_idx], v2[hist_idx], v3[hist_idx], refres_images);
    }
    return merge_histograms(hists, *this, true);
  }

  //////////////////////////////////////////////////////////////////////////////

  /*! Templated constructor for 4 vectors parameters.
   *  Making all histograms separately then merging them.
   * \see function factory create() for all possibilites of arguments */
  template<class _T1, class _T2, class _T3, class _T4>
  bool create (const std::vector<_T1> & v1,
               const std::vector<_T2> & v2,
               const std::vector<_T3> & v3,
               const std::vector<_T4> & v4,
               bool want_refresh_illus_images = true) {
    DEBUG_PRINT("PersonHistogram::create(line %i)\n", __LINE__);
    clear();
    unsigned int nhists = v1.size();
    std::vector<PersonHistogram> hists(nhists, PersonHistogram());
    for (unsigned int hist_idx = 0; hist_idx < nhists; ++hist_idx){
      // only refresh for last image (used for merge)
      bool refres_images = want_refresh_illus_images && (hist_idx == nhists - 1);
      hists[hist_idx].create(v1[hist_idx], v2[hist_idx], v3[hist_idx], v4[hist_idx], refres_images);
    }
    return merge_histograms(hists, *this, true);
  }

  //////////////////////////////////////////////////////////////////////////////

  //! dtor
  virtual ~PersonHistogram() {}

  //////////////////////////////////////////////////////////////////////////////

  /*!
   * Estimate how an histogram is similar to another one.
   * \param ph2
   *    another histogram to compare
   * \param method
   *    the computation method, cf cv::compareHist()
   * \return 0 for identical histograms
   *    and 1 for completely different histograms
 */
  inline double compare_to(const PersonHistogram & ph2,
                           const int method = CV_COMP_BHATTACHARYYA) const {
    return histogram_utils::distance_hist_vectors(_hist_vector, ph2._hist_vector, method);
  }

  //////////////////////////////////////////////////////////////////////////////

  /*!
   * Save to a stream.
   * CF http://docs.opencv.org/doc/tutorials/core/file_input_output_with_xml_yml/file_input_output_with_xml_yml.html
   * \param fs
   * \param save_illus_images
   */
  inline void write(cv::FileStorage &fs) const {
    fs << "hists" << "[";
    for( unsigned int i = 0; i < _hist_vector.size(); i++ )
      fs << _hist_vector[i];
    fs << "]";
    fs << "hist_input_images_nb" << _input_images_nb;
    fs << "illus_color_img" << _illus_color_img;
  } // end write()

  //////////////////////////////////////////////////////////////////////////////

  static inline bool is_bg(const cv::Vec3b & val) {
    return val[0] == PH_ILLUS_BG_COLOR_B
        && val[1] == PH_ILLUS_BG_COLOR_G
        && val[2] == PH_ILLUS_BG_COLOR_R;
}

  inline void read(const cv::FileNode &fn) {
    fn["illus_color_img"] >> _illus_color_img;
    // regenerate mask
    if (!_illus_color_img.empty())
      // image_utils::mask(_illus_color_img, _user_mask_bbox, image_utils::is_zero_vec3b);
      image_utils::mask(_illus_color_img, _user_mask_bbox, PersonHistogram::is_bg);
    fn["hist_input_images_nb"] >> _input_images_nb;
    cv::FileNode hist_nodes = fn["hists"];
    _hist_vector.resize(hist_nodes.size());
    for (unsigned int hist_idx = 0; hist_idx < hist_nodes.size(); ++hist_idx)
      hist_nodes[hist_idx] >> _hist_vector[hist_idx];
  } // end read

  ////////////////////////////////////////////////////////////////////////////////

  static inline bool merge_histograms(const std::vector<PersonHistogram> & phists,
                                      PersonHistogram & out,
                                      bool want_normalize_hist = true) {
    printf("merge_histograms(%i histograms)\n", phists.size());
    if (phists.size() == 0) {
      out = PersonHistogram();
      return true;
    }
    out = PersonHistogram(phists.back());
    unsigned int npersons = phists.size();
    HistVec hists;
    hists.resize(npersons);
    std::vector<double> weights;
    weights.resize(npersons);
    // run through all body parts
    for (unsigned int hist_idx = 0; hist_idx < BODY_PARTS; ++hist_idx) {
      // run through all persons
      for (unsigned int person_idx = 0; person_idx < npersons; ++person_idx) {
        hists[person_idx] = phists[person_idx]._hist_vector[hist_idx];
        weights[person_idx] = phists[person_idx]._input_images_nb;
        //  printf("person_idx:%i, hist_idx:%i, nbins:%i\n",
        //         person_idx, hist_idx, hists[person_idx].rows);
      } // end loop person_idx
      bool ok = histogram_utils::merge_histograms(hists, weights, out._hist_vector[hist_idx],
                                                  want_normalize_hist);
      if (!ok)
        return false;
    } // end loop hist_idx
    out._input_images_nb = std::accumulate(weights.begin(), weights.end(), 0.);
    out.refresh_illus_images_frame_counter();
    return true;
  } // end merge_histograms();

  //////////////////////////////////////////////////////////////////////////////

  bool to_mat(cv::Mat & out) const {
    int ncols = MAT_COLS, nrows = 1;
    if (out.empty())
      out.create(nrows, ncols, CV_32FC1);
    if (out.cols != ncols || out.rows != nrows || out.type() != CV_32FC1) {
      printf("to_mat(): out has not the correct size:'%s' != (%ix%i, type:CV_32FC1)\n",
             image_utils::infosImage(out).c_str(), ncols, nrows);
      return false;
    }

#ifdef SVM_USE_STD_DEV_HIST
    if (_hist_vector.size() != BODY_PARTS) {
      printf("to_mat(): _hist_vector has not the correct size:%i != BODY_PARTS %i\n",
             _hist_vector.size(), BODY_PARTS);
      return false;
    }

    for (unsigned int hist_idx = 0; hist_idx < BODY_PARTS; ++hist_idx) {
      const histogram_utils::Histogram* hist = &(_hist_vector.at(hist_idx));
      int nbins = hist->rows;
      if (histogram_utils::is_histogram_empty(*hist) || nbins != HIST_NBINS) {
        printf("to_mat(): hist '%s', nbins()=%i != ncols=%i\n",
               histogram_utils::hist_to_string(*hist).c_str(), nbins, ncols);
        return false;
      }
      double mean, std_dev;
      histogram_utils::mean_std_dev_modulo(*hist, 180, mean, std_dev);
      out.at<float>(0, 2*hist_idx)   = mean;
      out.at<float>(0, 2*hist_idx+1) = std_dev;
    } // end loop hist_idx
#else // not SVM_USE_STD_DEV_HIST
    const histogram_utils::Histogram* hist = &(_hist_vector.at(1));
    int nbins = hist->rows;
    if (nbins != ncols) {
      printf("to_mat(): nbins()=%i != ncols=%i\n", nbins, ncols);
      return false;
    }
    // image.at<uchar>(row,col)
    for (unsigned int bin_idx = 0; bin_idx < nbins; ++bin_idx)
      out.at<float>(0, bin_idx) = hist->at<float>(bin_idx);
#endif // SVM_USE_STD_DEV_HIST
    return true;
  }

  //////////////////////////////////////////////////////////////////////////////

  inline void to_illus_image(cv::Mat3b & out) const {
    histogram_utils::vector_of_histograms_to_image
        (get_hist_vector(), out, 300, 200, colormaps::ratio2hue);
  }

  inline void show_illus_image(int delay = -1) const {
    cv::Mat3b illus;
    to_illus_image(illus);
    cv::imshow("illus", illus);
    if (delay >= 0)
      cv::waitKey(delay);
  }

  //////////////////////////////////////////////////////////////////////////////

  //! \return a sentence describing the color of the clothes worn by the user
  inline std::string description_sentence(int user_index = 0) const {
    // printf("description_sentence(user_index:%i)\n", user_index);
    std::ostringstream out;
    if (_hist_vector.size() != 3) {
      out << "Incorrect size of the histogram vector of user " << (user_index + 1);
      return out.str();
    }
    // add dominant color for the torso, skip first bin (red of the skin)
    std::string top_color = histogram_utils::hue_hist_dominant_color_to_string(_hist_vector[1], true);
    bool top_color_found = (top_color.find("empty") == std::string::npos);
    // add dominant color for the legs, skip first bin (red of the skin)
    std::string trousers_color = histogram_utils::hue_hist_dominant_color_to_string(_hist_vector[2], true);
    bool trousers_color_found = (trousers_color.find("empty") == std::string::npos);

    if (!top_color_found && !trousers_color_found) {
      out << "I can't see what user " << (user_index + 1) << " wears ";
      return out.str();
    }
    out << "User " << (user_index + 1) << " wears ";
    if (top_color_found)
      out << "a " << top_color <<  " top";

    if (trousers_color_found) {
      if (top_color_found)
        out << " and ";
      out << trousers_color << " trousers";
    } // end if (trousers_color_found)
    return out.str();
  } // end description_sentence()

  //////////////////////////////////////////////////////////////////////////////

  inline const cv::Mat3b        & get_illus_color_img()  const { return _illus_color_img; }
  inline       cv::Mat3b        & get_illus_color_img()        { return _illus_color_img; }
  inline const cv::Mat1b        & get_illus_color_mask() const { return _user_mask_bbox; }
  inline       cv::Mat1b        & get_illus_color_mask()       { return _user_mask_bbox; }
  inline const cv::Mat1b        & get_multimask()        const { return _multimask; }
  inline const HistVec          & get_hist_vector()      const { return _hist_vector; }
  inline       int                get_input_images_nb()  const { return _input_images_nb; }
  inline const SEEN_BUFFER_TYPE & get_seen_buffer()      const { return _seen_buffer_short; }

  inline cv::Mat3b seen_buffer2img() const {
    cv::Mat1f seen_buffer_float_buffer;
    cv::Mat3b seen_buffer_illus;
    image_utils::propagative_floodfill_seen_buffer_to_viz_image
        (_seen_buffer_short, seen_buffer_float_buffer, seen_buffer_illus);
    cv::circle(seen_buffer_illus, _seed - _user_mask_bbox_rect.tl(), 4, CV_RGB(0, 0, 255), 1);
    cv::circle(seen_buffer_illus, _top_centered_seed, 4, CV_RGB(0, 0, 255), 2);
    return seen_buffer_illus;
  }

  //////////////////////////////////////////////////////////////////////////////
private:

  void clear() {
    _input_images_nb = 0;
    _hist_vector.resize(BODY_PARTS);
    _illus_color_img.create(1, 1);
    // _user_mask.create(1, 1); // no need to clear it
  }

  //////////////////////////////////////////////////////////////////////////////

  /*!
   * Determines the user mask from a known seed with the following steps:
   * 1) a depth Canny to find edges in the depth image
   * 2) a floodfill propagation in the edge image, starting from seed.
   * \param depth
   *    The depth image coming from a kinect.
   * \param seed
   *    a pixel (x, y) in rgb where the user is supposed to be.
   *    Used for the user mask computation (floodfilling fro this seed in edge image).
   */
  bool compute_user_mask(const cv::Mat1f & depth, const cv::Point & seed) {
    DEBUG_PRINT("compute_user_mask()\n");
    TIMER_RESET(_depth_canny.timer);
#if 0
    return _segmenter.find_blob
        (depth, seed, _user_mask, BlobSegmenter::FLOODFILL_EDGE_CLOSER, NULL, true,
         DepthCanny::DEFAULT_CANNY_THRES1, DepthCanny::DEFAULT_CANNY_THRES2,
         GroundPlaneFinder::DEFAULT_DISTANCE_THRESHOLD_M,
         GroundPlaneFinder::DEFAULT_LOWER_RATIO_TO_USE,
         image_utils::FloodFillEdgeCloser::DEFAULT_PREV_LINE_DIFF_THRES,
         image_utils::FloodFillEdgeCloser::DEFAULT_SRC_WIDTH_RATIO_THRES);
#else
    return _segmenter.find_blob
        (depth, seed, _user_mask, BlobSegmenter::GROUND_PLANE_FINDER);
#endif
  }

  //////////////////////////////////////////////////////////////////////////////

  /*!
   * \brief refresh_illus_images
   */
  inline void refresh_illus_images_frame_counter() {
    // printf("refresh_illus_images_frame_counter(%i)\n", _input_images_nb);
    // write nb of images
    if (_input_images_nb != 1) {
      cv::Rect text_roi = image_utils::putTextBackground
                          (_illus_color_img, StringUtils::cast_to_string(_input_images_nb),
                           cv::Point(_illus_color_img.cols / 2, _illus_color_img.rows - 5),
                           CV_FONT_HERSHEY_PLAIN, 2.f,
                           CV_RGB(255, 255, 255), CV_RGB(120, 0, 0), 1, 2);
      _user_mask_bbox(text_roi).setTo(255);
    }
    //    cv::imshow("_illus_color_img", _illus_color_img);
    //    cv::imshow("_user_mask_bbox", _user_mask_bbox);
    //    cv::waitKey(0);
  } // end refresh_illus_images_frame_counter();

  ////////////////////////////////////////////////////////////////////////////////

  // consistent data:
  HistVec _hist_vector;
  cv::Mat3b _illus_color_img;
  // cv::Mat1b _illus_mask_img;
  int _input_images_nb;
  cv::Mat1b _multimask;
  cv::Mat1b _user_mask_bbox;

  // temp data:
  BlobSegmenter _segmenter;
  cv::Mat1b _user_mask;
  cv::Mat3b _rgb_hsv;
  cv::Mat1b _hue;
  // vector of histograms
  cv::Point _seed, _top_centered_seed;
  cv::Rect _user_mask_bbox_rect;
  SEEN_BUFFER_TYPE _seen_buffer_short;
  cv::Mat1f _lookup_result;
};

#endif // PERSON_HISTOGRAM_H
