/*!
  \file        hist_tracking_skill.h
  \author      Arnaud Ramey <arnaud.a.ramey@gmail.com>
                -- Robotics Lab, University Carlos III of Madrid
  \date        2012/12/28

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

\class HistTrackingSkill
\brief A template for a cass that recognizes people according to the clothes they wear.
A current \a PersonHistogramSet is compared against a reference one
(loaded at startup), and the results are said with ETTS system,
using a \a EttsApi.

To use it, the population of the current \a PersonHistogramSet
(called \var curr_phset )
must be computed from outside.
A bunch of functions doing that with different kind of inputs are available,
\see documentation of \a PersonHistogram

\section Parameters

\section Subscriptions

\section Publications
 */

#ifndef HIST_TRACKING_SKILL_H
#define HIST_TRACKING_SKILL_H

// AD
#include "vision_utils/color_utils.h"
#include "vision_utils/utils/timer.h"
#include "std_msgs/String.h"
#include "vision_utils/nano_etts_api.h"
#include "vision_utils/test_person_histogram_set_variables.h"
#include "vision_utils/content_processing.h"
#include "vision_utils/user_image_to_rgb.h"
// people_msgs_rl
#include "people_recognition_vision/person_histogram_set.h"

class HistTrackingSkill {
public:
  typedef int ButtonIdx;
  static const ButtonIdx NO_USER_SELECTED = -1;
  typedef PersonHistogramSet::PersonLabel PersonLabel;

  HistTrackingSkill() {
    ros::NodeHandle _nh_public;
    _etts_api.advertise();
    DEBUG_PRINT("ctor HistTrackingSkill()");
    // init etts_api
    //_etts_api.setLanguage(Translator::LANGUAGE_ENGLISH);
    _etts_api.say_text("|en:Show me what clothes you wear!");

    // create reference PersonHistogramSet
    //    ref_phset.create
    //        (test_person_histogram_set_variables::all_filename_prefixes_struct(),
    //         test_person_histogram_set_variables::all_seeds_struct(),
    //         test_person_histogram_set_variables::all_kinect_serials_struct());
    ref_phset.push_back_vec
        (test_person_histogram_set_variables::refset_filename_prefixes,
         test_person_histogram_set_variables::refset_seeds,
         test_person_histogram_set_variables::refset_kinect_serials,
         test_person_histogram_set_variables::refset_labels());
    previous_phset_size = 0;

    // draw caption
    colormap_to_caption_image(_colormap_caption,
                              150, 300,
                              colormaps::ratio2red_green_half,
                              0., .6, .05, .1);

    // GUI
    ref_phs_prev_button = NO_USER_SELECTED;
    curr_phs_prev_button = NO_USER_SELECTED;
    int cols = 100, rows = 200;
    cv::Point center(cols/2, rows/2);
    new_ph_button.get_illus_color_img().create(rows, cols); // rows, cols
    new_ph_button.get_illus_color_img().setTo(200);
    new_ph_button.get_illus_color_mask().create(rows, cols); // rows, cols
    new_ph_button.get_illus_color_mask().setTo(255);
    image_utils::draw_text_centered
        (new_ph_button.get_illus_color_img(), "[NEW]",
         center, CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 0, 0));
    erase_ph_button.get_illus_color_img().create(rows, cols);
    erase_ph_button.get_illus_color_img().setTo(200);
    erase_ph_button.get_illus_color_mask().create(rows, cols); // rows, cols
    erase_ph_button.get_illus_color_mask().setTo(255);
    image_utils::draw_text_centered
        (erase_ph_button.get_illus_color_img(), "[erase]",
         center, CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 0, 0));
    repaint_ref_phset_illus();

    _interface_window_name = "interface";
    cv::namedWindow(_interface_window_name);
    cv::setMouseCallback(_interface_window_name, interface_mouse_cb, this);
    DEBUG_PRINT("end ctor HistTrackingSkill()");
  } // end ctor

  //////////////////////////////////////////////////////////////////////////////

  bool add_user_with_new_label(const ButtonIdx curr_button) {
    PersonLabel curr_label;
    if (!get_curr_label(curr_button, curr_label))
      return false;
    const PersonHistogram* curr_ph = NULL;
    if (!curr_phset.last_at(curr_label, curr_ph))
      return false;

    PersonLabel new_label = ref_phset.maxlabel() + 1;
    if (!ref_phset.push_back(*curr_ph, new_label))
      return false;
    ref_phset.save_back_up_file();
    return true;
  }

  //////////////////////////////////////////////////////////////////////////////

  bool add_user_with_label(const ButtonIdx ref_button, const ButtonIdx curr_button) {
    PersonLabel ref_label, curr_label;
    if (!button2label_ref(ref_button, ref_label))
      return false;
    if (!get_curr_label(curr_button, curr_label))
      return false;
    const PersonHistogram* curr_ph = NULL;
    if (!curr_phset.last_at(curr_label, curr_ph))
      return false;
    // now really add it
    if (!ref_phset.push_back(*curr_ph, ref_label))
      return false;
    ref_phset.save_back_up_file();
    return true;
  }

  //////////////////////////////////////////////////////////////////////////////

  bool erase_user(const ButtonIdx ref_button) {
    PersonLabel ref_label;
    if (!button2label_ref(ref_button, ref_label))
      return false;
    // erase
    if (!ref_phset.erase_by_label(ref_label))
      return false;
    ref_phset.save_back_up_file();
    return true;
  }

  //////////////////////////////////////////////////////////////////////////////



  /*!
   * Compare the reference PersonHistogramSet to the current one, \a curr_phset.
   * As such, \a curr_phset should be populated before calling \a compare().
   */
  virtual bool compare() {
    printf("compare(ref:%i, curr:%i)\n", nusers_ref(), nhists_curr());
    DEBUG_PRINT("compare() - compareTo()");
    bool ok = ref_phset.compare_to(curr_phset,
                                   _best_assign,
                                   true, // generate_dist_matrix_illus
                                   false, // generate_dist_matrix_illus_caption,
                                   CV_COMP_BHATTACHARYYA,
                                   false); // are_vector_equals
    if (!ok)
      return false;
    ref_phset.assign2labels(_best_assign, _best_label_assign);
    DEBUG_PRINT("compare() - repaint_curr_phset_illus()");
    repaint_curr_phset_illus();

    ROS_WARN_THROTTLE(1, "assign:%s, dist_matrix:'%s'",
                      assignment_utils::assignment_list_to_string(_best_assign).c_str(),
                      ref_phset.get_dist_matrix().to_string(15).c_str());
    return true;
  } // end compare()

  //////////////////////////////////////////////////////////////////////////////

  /*!
   * make the illustrations images after calling \a compare().
   */
  virtual void illus(bool illus_interface = true,
                     bool illus_dist_matrix_illus = true,
                     bool illus_colormap_caption = true,
                     bool illus_ref_phset = true,
                     bool illus_curr_phset = true,
                     bool illus_rgb_per_user = true,
                     bool illus_hist_per_user = true,
                     bool illus_seen_buffer_per_user = true,
                     bool illus_multimask_illus_per_user = true) {
    DEBUG_PRINT("illus()");
    if (curr_phset.nhists() > 0)
    { TIMER_DISPLAY_CHART(curr_phset.front()._depth_canny.timer, 1); }

    // destroy unneeded windows
    for (unsigned int ph_idx = curr_phset.nhists(); ph_idx < previous_phset_size; ++ph_idx) {
      std::ostringstream window_title;
      window_title.str("");  window_title << "hist_illus #" << (int) ph_idx;
      cv::destroyWindow(window_title.str());
      window_title.str("");  window_title << "multimask #" << (int) ph_idx;
      cv::destroyWindow(window_title.str());
    } // end loop ph
    previous_phset_size = curr_phset.nhists();

    // histogram illus
    // printf("first loop for illustrations (0..%i)\n", phset.nhists());
    std::vector<PersonLabel> unique_labels = curr_phset.get_unique_labels();
    for (unsigned int label_idx = 0; label_idx < nusers_curr(); ++label_idx) {
      PersonLabel curr_label = unique_labels[label_idx];
      const PersonHistogram* ph;
      if (!curr_phset.last_at(curr_label, ph))
        continue;
      std::ostringstream window_title;

      // user color mask illus
      if (illus_rgb_per_user && !ph->get_illus_color_img().empty()) {
        window_title.str(""); window_title << "illus_color_img #" << (int) curr_label;
        cv::imshow("illus_color_img", ph->get_illus_color_img());
      } // end if (illus_rgb_per_user)

      if (illus_hist_per_user) {
        ph->to_illus_image(hist_illus);
        if (!hist_illus.empty()) {
          window_title.str("");  window_title << "hist_illus #" << (int) curr_label;
          cv::imshow(window_title.str(), hist_illus);
        }
      } // end if (illus_hist_per_user)

      if (illus_seen_buffer_per_user) {
        image_utils::propagative_floodfill_seen_buffer_to_viz_image
            (ph->get_seen_buffer(), seen_buffer_float_buffer, seen_buffer_illus);
        if (!seen_buffer_illus.empty()) {
          window_title.str("");  window_title << "seen_buffer #" << (int) curr_label;
          cv::imshow(window_title.str(), seen_buffer_illus);
        }
      } // end if (illus_seen_buffer_per_user)

      if (illus_multimask_illus_per_user) {
        user_image_to_rgb(ph->get_multimask(), multimask_illus, 8);
        if (!multimask_illus.empty()) {
          window_title.str("");  window_title << "multimask #" << (int) curr_label;
          cv::imshow(window_title.str(), multimask_illus);
        }
      } // end if (illus_multimask_illus_per_user)
    } // end loop label_idx

    if (illus_interface && !_interface.empty())
      cv::imshow(_interface_window_name, _interface);
    if (illus_dist_matrix_illus && !ref_phset.get_dist_matrix_illus().empty())
      cv::imshow("dist_matrix_illus", ref_phset.get_dist_matrix_illus());
    if (illus_colormap_caption && !_colormap_caption.empty())
      cv::imshow("dist_matrix_colormap_caption", _colormap_caption);
    if (illus_ref_phset && !_ref_phset_illus.empty())
      cv::imshow("ref_phset_illus", _ref_phset_illus);
    if (illus_curr_phset && !_curr_phset_illus.empty())
      cv::imshow("curr_phset_illus", _curr_phset_illus);
  } // end illus()

  //////////////////////////////////////////////////////////////////////////////

  void say_description_sentence(const int sentence_timeout = 15) {
    DEBUG_PRINT("say_description_sentence()");
    if (sentence_time.getTimeSeconds() < sentence_timeout)
      return;
    // build the sentences to say
    std::vector<std::string> sentences_to_say;
    unsigned int n_users = nusers_curr();
    if (n_users == 0) {
      sentences_to_say.push_back("|en:I cant see anybody around. Come and play!");
    } else { // n_users > 0
      std::vector<PersonLabel> unique_labels = curr_phset.get_unique_labels();
      for (unsigned int label_idx = 0; label_idx < curr_phset.nhists(); ++label_idx) {
        PersonLabel curr_label = unique_labels[label_idx];
        std::ostringstream sentence_str;
        const PersonHistogram* ph;
        if (!curr_phset.last_at(curr_label, ph))
          continue;
        sentence_str << "|en:" << ph->description_sentence(curr_label);
        sentences_to_say.push_back(sentence_str.str());
      } // end loop label_idx
    } // end if (n_users > 0)

    // only say them if never said before
    for (unsigned int sentence_idx = 0; sentence_idx < sentences_to_say.size(); ++sentence_idx) {
      std::string sentence = sentences_to_say[sentence_idx];
      if (std::find(last_sentences_said.begin(), last_sentences_said.end(),
                    sentence) != last_sentences_said.end()) {
        maggieDebug2("Sentence '%s' was already said, skipping it.",
                     sentence.c_str());
        continue;
      }
      _etts_api.say_text(sentence);
    } // end loop sentence

    // reset last sentences
    last_sentences_said = sentences_to_say;
    sentence_time.reset();
  } // say_description_sentence say();

  //////////////////////////////////////////////////////////////////////////////

  inline unsigned int nusers_ref()  const { return ref_phset.nlabels(); }
  inline unsigned int nhists_ref()  const { return ref_phset.nhists(); }
  inline unsigned int nusers_curr() const { return curr_phset.nlabels(); }
  inline unsigned int nhists_curr() const { return curr_phset.nhists(); }

  //////////////////////////////////////////////////////////////////////////////

  //! \return the reference PersonHistogramSet
  inline const PersonHistogramSet & get_ref_phset()  const {return ref_phset; }
  inline       PersonHistogramSet & get_ref_phset()        {return ref_phset; }
  //! \return the current PersonHistogramSet
  inline const PersonHistogramSet & get_curr_phset() const {return curr_phset; }
  inline       PersonHistogramSet & get_curr_phset()       {return curr_phset; }

  inline PersonHistogramSet::PHMatch    get_best_assign()       const { return _best_assign; }
  inline PersonHistogramSet::LabelMatch get_best_label_assign() const { return _best_label_assign; }

protected:

  //////////////////////////////////////////////////////////////////////////////

  bool button2label_ref(const ButtonIdx ref_button, PersonLabel & ref_label) {
    if (ref_button < 0 || ref_button >= (int) nusers_ref()) {
      printf("Can't find ref user at button %i, we have %i ref users\n",
             ref_button, nusers_ref());
      return false;
    }
    ref_label = ref_phset.get_unique_labels().at(ref_button);
    return true;
  }

  //////////////////////////////////////////////////////////////////////////////

  inline bool label2button_ref(const PersonLabel ref_label, ButtonIdx & ref_button) const {
    std::vector<PersonHistogramSet::PersonLabel> labels = ref_phset.get_unique_labels();
    for (ref_button = 0; ref_button < (int) labels.size(); ++ref_button) {
      if (labels[ref_button] == ref_label)
        return true;
    } // end for (label_idx)
    return false;
  }
  inline bool label2button_curr(const PersonLabel curr_label, ButtonIdx & curr_button) const {
    std::vector<PersonHistogramSet::PersonLabel> labels = curr_phset.get_unique_labels();
    for (curr_button = 0; curr_button < (int) labels.size(); ++curr_button) {
      if (labels[curr_button] == curr_label)
        return true;
    } // end for (label_idx)
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////

  bool get_curr_label(const ButtonIdx curr_button, PersonLabel & curr_label) {
    if (curr_button < 0 || curr_button >= (int) nusers_curr()) {
      printf("Can't find curr user at button %i, we have %i curr users\n",
             curr_button, nusers_curr());
      return false;
    }
    // get label
    if (curr_button >= (int) nusers_curr()) {
      printf("curr button %i greater than nb of labels (%i)\n", curr_button, nusers_curr());
      return false;
    }
    curr_label = curr_phset.get_unique_labels().at(curr_button);
    return true;
  }

  //////////////////////////////////////////////////////////////////////////////

  static void interface_mouse_cb(int event, int x, int y, int flags, void* userdata) {
    // ROS_WARN("interface_mouse_cb(x:%i, y:%i)", x, y);
    if (event != cv::EVENT_LBUTTONDOWN) // needs left click
      return;

    HistTrackingSkill* this_ptr = (HistTrackingSkill*) userdata;
    bool clicked_on_ref = (y < this_ptr->_interface.cols / 2);
    if (clicked_on_ref) { // first half -> ref phset
      ButtonIdx ref_button_idx = image_utils::paste_images_pixel_belong_to_image
          (x, y, true,
           PersonHistogramSet::CAPTION_WIDTH1,
           PersonHistogramSet::CAPTION_HEIGHT1, true);
      ButtonIdx curr_button_idx = this_ptr->curr_phs_prev_button;
      int ref_imgs = this_ptr->nusers_ref();
      if (ref_button_idx < 0 || ref_button_idx >= ref_imgs+2) { // add buttons
        ROS_WARN("User click (x:%i, y:%i), not one of the items of the ref interface: "
                 "ref_button_idx:%i, curr_button_idx:%i",
                 x, y, ref_button_idx, curr_button_idx);
        if (this_ptr->ref_phs_prev_button != NO_USER_SELECTED) {
          this_ptr->ref_phs_prev_button = NO_USER_SELECTED; // unselecting
          this_ptr->repaint_interface();
        }
        return;
      }

      // Buttons in this order: <histograms> [erase] [NEW]
      bool erase_pressed = (ref_button_idx == ref_imgs);
      bool new_pressed = (ref_button_idx == ref_imgs + 1);
      if (erase_pressed) {
        ROS_WARN("Deleting reference PH with label#%i", this_ptr->ref_phs_prev_button);
        this_ptr->erase_user(this_ptr->ref_phs_prev_button);
        this_ptr->ref_phs_prev_button = NO_USER_SELECTED; // unselecting
        this_ptr->curr_phs_prev_button = NO_USER_SELECTED; // also unselecting current PH
        this_ptr->repaint_ref_phset_illus();
      } // end if (erase_pressed)

      else if (new_pressed) {
        ROS_WARN("Adding current PH with label #%i with a new label", curr_button_idx);
        this_ptr->add_user_with_new_label(curr_button_idx);
        this_ptr->ref_phs_prev_button = NO_USER_SELECTED; // unselecting
        this_ptr->curr_phs_prev_button = NO_USER_SELECTED; // also unselecting current PH
        this_ptr->repaint_ref_phset_illus();
      } // end if (new_pressed)

      // a PH selectionned in curr PhSet -> create with given label
      else if (curr_button_idx != NO_USER_SELECTED) {
        ROS_WARN("Adding current PersonHistogram #%i to reference set with ref label #%i",
                 curr_button_idx, ref_button_idx);
        this_ptr->add_user_with_label(ref_button_idx, curr_button_idx);
        this_ptr->ref_phs_prev_button = NO_USER_SELECTED; // unselecting
        this_ptr->curr_phs_prev_button = NO_USER_SELECTED; // also unselecting current PH
        this_ptr->repaint_ref_phset_illus();
      } // end if not (new_pressed)

      else if (ref_button_idx == this_ptr->ref_phs_prev_button) {
        ROS_WARN("Unselecting reference PH with label#%i", ref_button_idx);
        this_ptr->ref_phs_prev_button = NO_USER_SELECTED; // unselecting
        this_ptr->repaint_interface();
      }

      else { // selecting a ref user
        ROS_WARN("Selecting reference PH with label#%i", ref_button_idx);
        this_ptr->ref_phs_prev_button = ref_button_idx; // storing selection
        this_ptr->repaint_interface();
      }

    } // end clicked_on_ref
    else { // 2nd half -> curr phset
      y = y - this_ptr->_ref_phset_illus.rows; // offset due to ref phset being above
      ButtonIdx ref_button_idx = this_ptr->ref_phs_prev_button;
      ButtonIdx curr_button_idx = image_utils::paste_images_pixel_belong_to_image
          (x, y, true,
           PersonHistogramSet::CAPTION_WIDTH1,
           PersonHistogramSet::CAPTION_HEIGHT1,
           true);
      if (curr_button_idx < 0 || curr_button_idx >= (int) this_ptr->nusers_curr()) {
        ROS_WARN("User click (x:%i, y:%i), not one of the items of the curr interface: "
                 "ref_button_idx:%i, curr_button_idx:%i",
                 x, y, ref_button_idx, curr_button_idx);
        if (this_ptr->curr_phs_prev_button != NO_USER_SELECTED) {
          this_ptr->curr_phs_prev_button = NO_USER_SELECTED; // unselecting
          this_ptr->repaint_interface();
        }
      }
      else if (curr_button_idx == this_ptr->curr_phs_prev_button) {
        ROS_WARN("Unselecting currerence PH with label#%i", curr_button_idx);
        this_ptr->curr_phs_prev_button = NO_USER_SELECTED; // unselecting
        this_ptr->repaint_interface();
      }

      else { // selecting a curr user
        ROS_WARN("Selecting currerence PH with label#%i", curr_button_idx);
        this_ptr->curr_phs_prev_button = curr_button_idx; // storing selection
        this_ptr->repaint_interface();
      }
    } // end clicked on curr
  } // end interface_mouse_cb();

  //////////////////////////////////////////////////////////////////////////////

  //! wait some time for repainting the images
  void wait_key() const {
    char c = cv::waitKey(25);
    if ((int) c == 27)
      exit(-1);
  } // end wait_key();

  //////////////////////////////////////////////////////////////////////////////

  inline void repaint_interface() {
    DEBUG_PRINT("repaint_interface(): ref:'%s', curr:'%s'",
                image_utils::infosImage(_ref_phset_illus).c_str(),
                image_utils::infosImage(_curr_phset_illus).c_str());
    int expected_img_rows = PersonHistogramSet::CAPTION_HEIGHT1 + image_utils::HEADER_SIZE;
    // copy paste _ref_phset_illus and _curr_phset_illus
    bool ref_img_good = _ref_phset_illus.rows == expected_img_rows && _ref_phset_illus.cols > 0;
    bool curr_img_good = _curr_phset_illus.rows == expected_img_rows && _curr_phset_illus.cols > 0;
    if (!ref_img_good && !curr_img_good) {
      printf("repaint_interface(): both curr and ref phset illus non valid! "
             "(ref:'%s', curr:'%s')\n",
             image_utils::infosImage(_ref_phset_illus).c_str(),
             image_utils::infosImage(_curr_phset_illus).c_str());
      _interface.create(expected_img_rows, 100);
      _interface.setTo(0);
      return;
    }
    if (!curr_img_good) {
      printf("repaint_interface(): only ref phset illus valid! "
             "(ref:'%s', curr:'%s')\n",
             image_utils::infosImage(_ref_phset_illus).c_str(),
             image_utils::infosImage(_curr_phset_illus).c_str());
      _ref_phset_illus.copyTo(_interface);
      return;
    }
    if (!ref_img_good) {
      printf("repaint_interface(): only curr phset illus valid!"
             "(ref:'%s', curr:'%s')\n",
             image_utils::infosImage(_ref_phset_illus).c_str(),
             image_utils::infosImage(_curr_phset_illus).c_str());
      _curr_phset_illus.copyTo(_interface);
      return;
    }
    // init image
    _interface.create(2*expected_img_rows,
                      std::max(_ref_phset_illus.cols, _curr_phset_illus.cols)); // rows, cols
    _interface.setTo(255);
    // define the ROIs
    cv::Mat3b inter_ref_roi = _interface(image_utils::bbox_full(_ref_phset_illus));
    cv::Mat3b inter_curr_roi = _interface(image_utils::bbox_full(_curr_phset_illus)
                                          + cv::Point(0, _ref_phset_illus.rows));
    // copy the interfaces
    _ref_phset_illus.copyTo(inter_ref_roi);
    _curr_phset_illus.copyTo(inter_curr_roi);

    // paint the selections
    if (ref_phs_prev_button != NO_USER_SELECTED) {
      cv::Rect ROI = image_utils::paste_images_image_roi
          (ref_phs_prev_button, true,
           PersonHistogramSet::CAPTION_WIDTH1,
           PersonHistogramSet::CAPTION_HEIGHT1,
           true);
      cv::rectangle(inter_ref_roi, ROI, CV_RGB(255, 0, 0), 3);
    }
    if (curr_phs_prev_button != NO_USER_SELECTED) {
      cv::Rect ROI = image_utils::paste_images_image_roi
          (curr_phs_prev_button, true,
           PersonHistogramSet::CAPTION_WIDTH1,
           PersonHistogramSet::CAPTION_HEIGHT1,
           true);
      cv::rectangle(inter_curr_roi, ROI, CV_RGB(255, 0, 0), 3);
    }

    // paint the best assign if found
    for (PersonHistogramSet::LabelMatch::const_iterator assign = _best_label_assign.begin();
         assign != _best_label_assign.end(); ++ assign) {
      ButtonIdx ref_button, curr_button;
      if (!label2button_ref(assign->first, ref_button)
          || !label2button_curr(assign->second, curr_button))
        continue;
      printf("Drawing lines between ref button %i and curr button %i\n",
             ref_button, curr_button);
      cv::Point ref_pt((.5+ref_button) * PersonHistogramSet::CAPTION_WIDTH1,
                       expected_img_rows * (1. - .2));
      cv::Point curr_pt((.5+curr_button) * PersonHistogramSet::CAPTION_WIDTH1,
                        expected_img_rows * (1. + .2));
      cv::line(_interface, ref_pt, curr_pt, CV_RGB(0, 255, 0), 3);
    } // end for assign
  } // end repaint_interface();

  //////////////////////////////////////////////////////////////////////////////

  inline void repaint_ref_phset_illus() {
    DEBUG_PRINT("repaint_ref_phset_illus()");
    // for ref_phset, add buttons [erase] and [NEW]
    _ref_phset_illus.release();
    ref_phset.push_back(erase_ph_button, 100001, false); // no retraining
    ref_phset.push_back(new_ph_button, 100002, false);
    ref_phset.generate_caption_image
        (_ref_phset_illus, &titlemaps::int_to_uppercase_letter);
    ref_phset.pop_back(false);
    ref_phset.pop_back(false);
    repaint_interface();
  } // end repaint_ref_phset_illus

  //////////////////////////////////////////////////////////////////////////////

  inline void repaint_curr_phset_illus() {
    DEBUG_PRINT("repaint_curr_phset_illus()");
    _curr_phset_illus.release();
    curr_phset.generate_caption_image(_curr_phset_illus, &titlemaps::int_to_lowercase_letter);
    repaint_interface();
  } // end repaint_curr_phset_illus

  //////////////////////////////////////////////////////////////////////////////

  // data
  cv::Mat3b hist_illus;
  cv::Mat1f seen_buffer_float_buffer;
  cv::Mat1b greyscale_buffer;
  cv::Mat3b seen_buffer_illus;
  cv::Mat3b multimask_illus;
  cv::Mat3b _colormap_caption;

  PersonHistogramSet curr_phset;
  unsigned int previous_phset_size;
  cv::Mat3b _curr_phset_illus;
  ButtonIdx curr_phs_prev_button;

  PersonHistogramSet ref_phset;
  cv::Mat3b _ref_phset_illus;
  PersonHistogram new_ph_button;
  PersonHistogram erase_ph_button;
  ButtonIdx ref_phs_prev_button;

  PersonHistogramSet::PHMatch _best_assign;
  PersonHistogramSet::LabelMatch _best_label_assign;
  std::string _interface_window_name;
  cv::Mat3b _interface;

  // stuff for etts
  NanoEttsApi _etts_api;
  Timer sentence_time;
  std::vector<std::string> last_sentences_said;
}; // end class HistTrackingSkill

#endif // HIST_TRACKING_SKILL_H
