// Dongzhuo Li 06/26/2018
#ifndef BND_H__
#define BND_H__

#include "Parameter.h"

class Bnd {
 private:
  int nz_, nx_, ny_, nPml_, nSteps_, nzBnd_, nxBnd_, nyBnd_, len_Bnd_vec_,
      nLayerStore_;

  bool isAc_, withAdj_;

 public:
  float *d_Bnd_vz, *d_Bnd_vx, *d_Bnd_vy, *d_Bnd_szz, *d_Bnd_sxz, *d_Bnd_sxx, *d_Bnd_syy, *d_Bnd_sxy, *d_Bnd_syz;

  Bnd(const Parameter &para);
  Bnd(const Bnd&) = delete;
  Bnd& operator=(const Bnd&) = delete;

  ~Bnd();

  void field_from_bnd(float *d_szz, float *d_sxz, float *d_sxx,
                      float *d_syy, float *d_sxy, float *d_syz,
                      float *d_vz, float *d_vx, float *d_vy, int indT);

  void field_to_bnd(float *d_szz, float *d_sxz, float *d_sxx,
                      float *d_syy, float *d_sxy, float *d_syz,
                      float *d_vz, float *d_vx, float *d_vy, int indT, bool if_stress);

  int len_Bnd_vec() { return len_Bnd_vec_; }
};

#endif