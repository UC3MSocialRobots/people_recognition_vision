/*!
  \file        height_detector.h
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2013/9/11

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

A class for computing the height of a user in a mask.

 */

#ifndef HEIGHT_DETECTOR_H
#define HEIGHT_DETECTOR_H

#include "vision_utils/kinect_openni_utils.h"
#include "vision_utils/image_comparer.h"
#include <vision_utils/img_path.h>
#include "vision_utils/head_finder.h"

class HeightDetector {
public:
  static const double CONFIDENCE_SCALE_FACTOR = 10;

  static const int    ERROR= -1;
  static const int    NOT_COMPUTED = -2;

  //! A mini class for storing results of the height detection
  struct Height {
    int height_px; // in pixels
    double height_m; // in meters
    double height_confidence; // 1: very similar to model, 0: completely different
    Height() : height_px (NOT_COMPUTED), height_m (NOT_COMPUTED), height_confidence (NOT_COMPUTED)
    {}
    inline std::string to_string() const {
      std::ostringstream ans;
      ans << height_px << "px";

      if (height_m == NOT_COMPUTED) ans << " (height_m not computed)";
      else if (height_m == ERROR)   ans << " (HEIGHT_M ERROR)";
      else                          ans << "=" << height_m << "m";

      if (height_confidence == NOT_COMPUTED) ans << " (confidence not computed)";
      else if (height_confidence == ERROR)   ans << " (CONFIDENCE ERROR)";
      else                            ans << " (" << (int) (100 * height_confidence) << "%sure)";
      return ans.str();
    }
    friend std::ostream& operator<< (std::ostream& stream, const Height& h) {
      stream << h.to_string();
      return stream;
    }
  }; // end struct Height

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  HeightDetector() {
    _ref_skel_filename = IMG_DIR "skeletons/ref_skel.png";
  }

  //////////////////////////////////////////////////////////////////////////////

  /*!
   * \brief height_pixels
   * \param user_mask
   * \return  -1 if error
   */
  Height height_pixels(const cv::Mat1b & user_mask,
                       bool compute_confidence = false) {
    Height herror;
    herror.height_px = ERROR;
    if (user_mask.empty()) {
      printf("height_pixels: emtpy mask, cant process it!\n");
      return herror;
    }
    if (!_head_finder.find(user_mask, _head_pos)) {
      maggiePrint("HeightDetector: could not find head pos!");
      return herror;
    }
    // point out of bbox (error in find_top_point_centered() for instance
    if (!image_utils::bbox_full(user_mask).contains(_head_pos)) {
      maggiePrint("HeightDetector: _head pos outside of user mask bounds!");
      return herror;
    }
    _skeleton_expanded.create(user_mask.size());
    _skeleton_expanded.setTo(0);
#if 0 // reuse skeleton of the head finder
    _head_finder.get_skeleton().copyTo(_skeleton_expanded(_head_finder.get_bbox()));
#else // recompute skeleton
    _thinner.thin(user_mask, _head_finder.get_thinning_method(), true);
    //  printf("skel:'%s', user_mask:'%s'\n",
    //         image_utils::infosImage(_thinner.get_skeleton()).c_str(),
    //         image_utils::infosImage(user_mask).c_str());
    _thinner.get_skeleton().copyTo(_skeleton_expanded(_thinner.get_bbox()));
    // head pos must be the closest point on _skeleton_expanded
    _head_pos = _head2closest_skeleton_finder.find(_skeleton_expanded, _head_pos);
    if (_head_pos.x < 0)
      return herror;
#endif
    // cv::imshow("_skeleton_expanded", _skeleton_expanded); cv::waitKey(0);

    // then find head point and
    // draw a vertical line from the seed to the edge of the mask
    while (_head_pos.y-1 >= 0
           && user_mask.at<uchar>(_head_pos.y-1, _head_pos.x) != 0) { // row, col
      // printf("_head_pos:(%i, %i)\n", _head_pos.x, _head_pos.y);
      _skeleton_expanded(_head_pos) = 255;
      --_head_pos.y;
    }
    // cv::imshow("skeleton_expanded", _skeleton_expanded); cv::waitKey(5);

    // then floodfill-propagate from head along skeletons
    // no need to search top point as it is already the one we have
    _filler.floodfill(_skeleton_expanded, _head_pos, false);
    cv::Mat1s* _seen_buffer = &_filler.get_queued();
    // image_utils::propagative_floodfill
    // (_skeleton_expanded, _head_pos, _seen_buffer, false);

    image_utils::BufferElem dist_head2feet;
    // find the remotest point at the level of the feet
    if (!image_utils::minnonzero_from_botton_row(*_seen_buffer, dist_head2feet, _foot_pos)) {
      return herror;
    }
    // printf("dist_head2feet:%i\n", dist_head2feet);

    // now draw a vertical line from foot to edge of mask
    while (_foot_pos.y+1 <= user_mask.rows - 1
           && user_mask.at<uchar>(_foot_pos.y+1, _foot_pos.x) != 0) {
      // printf("_foot_pos:(%i, %i)\n", _foot_pos.x, _foot_pos.y);
      ++_foot_pos.y;
      ++dist_head2feet; // augment the distance all along
      (*_seen_buffer)(_foot_pos) = dist_head2feet; // for a better visualization
      // skeleton_expanded(_foot_pos) = 255; // for debuggging, display skeleton_expanded then
    }
    // cv::imshow("_skeleton_expanded", _skeleton_expanded); cv::waitKey(0);
    Height ans;
    ans.height_px = (int) dist_head2feet;
    if (compute_confidence)
      ans.height_confidence = confidence_to_ref_skel();
    return ans;
  }

  //////////////////////////////////////////////////////////////////////////////

  //! \return height in meters or HEIGHT_METERS_ERROR if error
  Height height_meters(const cv::Mat1b & user_mask,
                       const double & pixel2meters_factor,
                       bool compute_confidence = false) {
    if (pixel2meters_factor == kinect_openni_utils::NAN_DOUBLE) {
      Height ans;
      ans.height_m = ERROR;
      return ans;
    }
    Height ans = height_pixels(user_mask, compute_confidence);
    if (ans.height_px < 0) {
      ans.height_m = ERROR;
      return ans;
    }
    ans.height_m = 1.f * ans.height_px * pixel2meters_factor;
    return ans;
  }

  //////////////////////////////////////////////////////////////////////////////

  //! \return height in meters or nan if error
  Height height_meters(const cv::Mat1f & depth,
                       const cv::Mat1b & user_mask,
                       const image_geometry::PinholeCameraModel & depth_camera_model,
                       bool compute_confidence = false) {
    if (user_mask.empty() || depth.size() != user_mask.size()) {
      printf("height_meters: dimensions of depth(%i, %i), "
             "user_mask(%i, %i) dont match!\n",
             depth.cols, depth.rows, user_mask.cols, user_mask.rows);
      Height ans;
      ans.height_m = ERROR;
      return ans;
    }
    double pixel2meters_factor = kinect_openni_utils::compute_average_pixel2meters_factor
                                 (depth, depth_camera_model, user_mask);
    // printf("pixel2meters_factor:%g\n", pixel2meters_factor);
    return height_meters(user_mask, pixel2meters_factor, compute_confidence);
  }


  //////////////////////////////////////////////////////////////////////////////

  /*!
   * \param user_mask
   *    the same image that was used for height_meters() or height_pixels()
   * \param out
   *    Need to be preallocuted, with a size of exactly
   *    (get_bbox().width, get_bbox().height)
   * \param thicker
   *    true for making spine thicker
   * \param h
   *    the measured height, its confidence is used to make
   *    the character more or less darker
   */
  void height2img(const cv::Mat1b & user_mask,
                  cv::Mat3b & out,
                  bool thicker = true,
                  Height h = Height(),
                  bool clear_before = true) {
    if (out.size() != user_mask.size())
      out.create(user_mask.size());

    // paint skeleton
    _filler.illus_img(out, clear_before);
    if (thicker) { // only dilate bbox
      cv::Rect bbox = image_utils::boundingBox(_filler.get_queued());
      if (bbox.width > 0 && bbox.height > 0) {
        cv::Mat3b out_bbox = out(bbox);
        cv::dilate(out_bbox, out_bbox, cv::Mat());
      }
    }

    // copy user mask where out is black
    double scale_factor = .5;
    if (h.height_confidence != ERROR && h.height_confidence != NOT_COMPUTED)
      scale_factor = clamp(2. * (h.height_confidence - .5), .3, 1.);
    cv::cvtColor(out, _out_bw, CV_BGR2GRAY);
    out.setTo(255 * scale_factor, user_mask & (_out_bw == 0));
    // cv::imshow("out", out); cv::waitKey(0);

    // draw head and feet
    cv::circle(out, _head_pos, 6, CV_RGB(0, 255, 0), 3);
    cv::circle(out, _foot_pos, 6, CV_RGB(0, 255, 0), 3);

    // write height on top of head
    cv::Point txt_pos = _head_pos + cv::Point(0, -5);
    std::ostringstream txt;
    if (h.height_m != ERROR && h.height_m != NOT_COMPUTED)
      txt << std::setprecision(3) <<  h.height_m << "m";
    else
      txt << std::setprecision(3) <<  h.height_px << "px";
    // txt << h.confidence << " lkl";
    cv::putText(out, txt.str(), txt_pos, CV_FONT_HERSHEY_PLAIN, 2,
                CV_RGB(0, 255, 0), 2);
    if (h.height_confidence != ERROR && h.height_confidence != NOT_COMPUTED) {
      txt.str(""); txt << (int) (100 *  h.height_confidence) << "%sure";
      cv::putText(out, txt.str(), txt_pos + cv::Point(0, -25),
                  CV_FONT_HERSHEY_PLAIN, 2, CV_RGB(0, 255, 0), 2);
    }

  } // end build_height_pixels_img();

  //////////////////////////////////////////////////////////////////////////////

  /*!
   * \brief height_meters_all_vales
   * \param depth
   * \param user_mask
   * \param depth_camera_model
   * \param min_idx, max_idx
   *   min and max indices in \ a user_mask (are included)
   * \return
   */
  bool height_meters_all_values(const cv::Mat1f & depth,
                                const cv::Mat1b & user_mask,
                                const image_geometry::PinholeCameraModel & depth_camera_model,
                                std::map<int, Height> & heights,
                                bool compute_confidence = false,
                                bool want_illus = false,
                                cv::Mat3b* user_skels_illus = NULL) {
    if (user_mask.empty() || depth.size() != user_mask.size()) {
      printf("height_meters_all_values: dimensions of depth(%i, %i), "
             "user_mask(%i, %i) dont match!\n",
             depth.cols, depth.rows, user_mask.cols, user_mask.rows);
      return false;
    }
    heights.clear();
    if (want_illus) {
      user_skels_illus->create(depth.size());
      user_skels_illus->setTo(0);
    }
    // find all users
    std::vector<uchar> user_indices;
    image_utils::get_all_different_values(user_mask, user_indices, true);
    unsigned int nusers = user_indices.size();

    // iterate on all user values
    cv::Mat1b curr_user_mask;
    for (unsigned int user = 0; user < nusers; ++user) {
      curr_user_mask = (user_mask == user_indices[user]);
      Height curr_height = height_meters(depth, curr_user_mask, depth_camera_model,
                                         compute_confidence);
      heights.insert(std::pair<int, Height>(user_indices[user], curr_height));

      if (want_illus && curr_height.height_m != ERROR) // only draw if success
        height2img(curr_user_mask, *user_skels_illus, true, curr_height, false);
    } // end loop user
    // dilate now
    return true;
  } // end height_meters_all_values()

  //////////////////////////////////////////////////////////////////////////////

  inline std::string get_ref_skel_filename() const {
    return _ref_skel_filename;
  }
  inline const cv::Mat1s & get_filler_queue() const { return _filler.get_queued(); }
  inline cv::Point get_head_pos() const { return _head_pos; }
  inline cv::Point get_foot_pos() const { return _foot_pos; }

protected:

  //////////////////////////////////////////////////////////////////////////////

  double confidence_to_ref_skel() {
    if (_skeleton_expanded.empty()) {
      printf("Empty skeleton_expanded, did u call height_pixels()?\n");
      return ERROR;
    }
    // load _ref_skel if needed
    if (_ref_skel_comparer.get_models_nb() != 1) {
      // load ref skeleton
      cv::Mat ref_skel = cv::imread(_ref_skel_filename, CV_LOAD_IMAGE_GRAYSCALE);
      ref_skel = (ref_skel == 255);
      // compute its voronoi
      VoronoiThinner ref_thinner;
      ref_thinner.thin(ref_skel, _head_finder.get_thinning_method(), true);
      // now setup ImageComparer
      std::vector<cv::Mat> model_as_vec(1, ref_thinner.get_skeleton());
      _ref_skel_comparer.set_models(model_as_vec, cv::Size(32, 32), true);
      if (_ref_skel_comparer.get_models_nb() != 1) {
        printf("Failed in loading ref skeleton model '%s'\n",
               _ref_skel_filename.c_str());
        return ERROR;
      }
      printf("Succesfully loaded ref skeleton model '%s':'%s'\n",
             _ref_skel_filename.c_str(),
             ""); //_ref_skel_comparer.model_to_string(0).c_str());
      // cv::imshow("ref_skel", ref_skel); cv::waitKey(0);
    } // end if (_ref_skel_comparer.get_models_nb() != 1)

    bool ok = _ref_skel_comparer.compareFile(_skeleton_expanded);
    if (!ok) {
      printf("Comparing with ref skeleton failed!\n");
      return ERROR;
    }
    return _ref_skel_comparer.get_confidence(CONFIDENCE_SCALE_FACTOR);
  } // end confidence_to_ref_skel()

  //////////////////////////////////////////////////////////////////////////////

  HeadFinder _head_finder;
  cv::Mat1b _skeleton_expanded;
  VoronoiThinner _thinner;
  image_utils::ClosestPointInMask2<uchar> _head2closest_skeleton_finder;

  // propagation stuff
  image_utils::PropagativeFloodfiller<uchar> _filler;
  cv::Point _head_pos, _foot_pos;

  // confidence stuff
  std::string _ref_skel_filename;
  ImageComparer _ref_skel_comparer;

  // viz stuff
  cv::Mat1b _out_bw;
  cv::Mat3b _viz_img;
};

#endif // HEIGHT_DETECTOR_H
