// Dongzhuo Li 04/20/2018
#ifndef MODEL_H__
#define MODEL_H__

#include "Parameter.h"

// void fileBinLoad(float *h_bin, int size, std::string fname);
// void intialArray(float *ip, int size, float value);
// __global__ void moduliInit(float *d_Vp, float *d_Vs, float *d_Rho, float
// *d_Lambda, float *d_Mu, int nx, int ny);

class Model {
 private:
  int nz_, nx_, ny_, nShape;

 public:
  Model();
  Model(const Parameter &para);
  Model(const Parameter &para, const float *Vp_, const float *Vs_,
        const float *Rho_);
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  ~Model();

  float *h_Vp;
  float *h_Vs;
  float *h_Rho;
  float *d_Vp;
  float *d_Vs;
  float *d_Rho;

  float *h_Lambda;
  float *d_Lambda;  
  float *h_Mu;
  float *d_Mu;
  float *d_Byc;

  float *h_LambdaGrad;
  float *d_LambdaGrad;
  float *h_MuGrad;
  float *d_MuGrad;
  float *h_RhoGrad;
  float *d_RhoGrad;

  int nz() { return nz_; }
  int nx() { return nx_; }
  int ny() { return ny_; }
};

#endif