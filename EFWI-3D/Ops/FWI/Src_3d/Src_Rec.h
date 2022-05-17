// Dongzhuo Li 05/13/2018
#ifndef SRC_REC_H__
#define SRC_REC_H__

#include <vector>
#include "Parameter.h"
#include "utilities.h"

class Src_Rec {
 private:
  bool if_res_;
  bool if_win_;

 public:
  std::vector<int> vec_z_src;
  std::vector<int> vec_x_src;
  std::vector<int> vec_y_src;
  std::vector<int> vec_nrec;
  std::vector<int *> d_vec_z_rec;
  std::vector<int *> d_vec_x_rec;
  std::vector<int *> d_vec_y_rec;
  std::vector<float *> d_vec_win_start;  // device side window
  std::vector<float *> d_vec_win_end;    // device side window
  std::vector<float *> d_vec_weights;    // weights for traces
  std::vector<float> vec_srcweights;     // weights for sources
  std::vector<float *> vec_source;
  std::vector<float *> d_vec_source;
  std::vector<float *> vec_data_x;
  std::vector<float *> vec_data_y;
  std::vector<float *> vec_data_z;
  std::vector<float *> vec_data_obs_x;
  std::vector<float *> vec_data_obs_y;
  std::vector<float *> vec_data_obs_z;
  std::vector<float *> vec_res_x;  // host side data residual
  std::vector<float *> vec_res_y;  // host side data residual
  std::vector<float *> vec_res_z;  // host side data residual
  int nShots;

  Src_Rec();
  Src_Rec(Parameter &para, std::string survey_fname);
  Src_Rec(Parameter &para, std::string survey_fname, const float *stf,
          int nShots, const int *shot_ids);
  Src_Rec(const Src_Rec&) = delete;
  Src_Rec& operator=(const Src_Rec&) = delete;
  ~Src_Rec();
};

#endif