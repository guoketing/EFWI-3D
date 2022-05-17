#include <iostream>
#include <string>
#include "Model.h"
#include "Parameter.h"
#include "utilities.h"

// model default constructor
Model::Model() {
  std::cout << "ERROR: You need to supply parameters to initialize models!"
            << std::endl;
  exit(1);
}

// model constructor from parameter file
Model::Model(const Parameter &para, const float *Lambda_, const float *Mu_, const float *Rho_) {
  nz_ = para.nz();
  nx_ = para.nx();
  ny_ = para.ny();
  nShape = nz_ * nx_ * ny_;

  const dim3 threads(32, 32, 1);
  int gridx = (nx_ + threads.x - 1) / threads.x;
  int gridy = (ny_ + threads.y - 1) / threads.y;
  int gridz = (nz_ + threads.z - 1) / threads.z;
  const dim3 blocks(gridx, gridy, gridz);

  h_Lambda = (float *)malloc(nShape * sizeof(float));
  h_Mu = (float *)malloc(nShape * sizeof(float));
  h_Rho = (float *)malloc(nShape * sizeof(float));
  h_Vp = (float *)malloc(nShape * sizeof(float));
  h_Vs = (float *)malloc(nShape * sizeof(float));
  h_LambdaGrad = (float *)malloc(nShape * sizeof(float));
  h_MuGrad = (float *)malloc(nShape * sizeof(float));
  h_RhoGrad = (float *)malloc(nShape * sizeof(float));

  for (int i = 0; i < nShape; i++) {
    if (Lambda_[i] < 0.0) {
      printf("Lambda is negative!!!");
      // exit(1);
    }
    h_Lambda[i] = Lambda_[i];
    h_Mu[i] = Mu_[i];
    h_Rho[i] = Rho_[i];
  }

  initialArray(h_Vp, nShape , 0.0);
  initialArray(h_Vs, nShape, 0.0);
  initialArray(h_LambdaGrad, nShape, 0.0);
  initialArray(h_MuGrad, nShape, 0.0);
  initialArray(h_RhoGrad, nShape, 0.0);

  CHECK(cudaMalloc((void **)&d_Lambda, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_Mu, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_Rho, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_Vp, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_Vs, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_Byc, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_LambdaGrad, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_MuGrad, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_RhoGrad, nShape * sizeof(float)));

  initialArrayGPU<<<blocks, threads>>>(d_LambdaGrad, nz_, nx_, ny_, 0.0);
  initialArrayGPU<<<blocks, threads>>>(d_MuGrad, nz_, nx_, ny_, 0.0);
  initialArrayGPU<<<blocks, threads>>>(d_RhoGrad, nz_, nx_, ny_, 0.0);
  initialArrayGPU<<<blocks, threads>>>(d_Byc, nz_, nx_, ny_, 1.0 / 1000.0);

  CHECK(cudaMemcpy(d_Lambda, h_Lambda, nShape * sizeof(float),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_Mu, h_Mu, nShape * sizeof(float),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_Rho, h_Rho, nShape * sizeof(float),
                   cudaMemcpyHostToDevice));


  velInit<<<blocks, threads>>>(d_Lambda, d_Mu, d_Rho, d_Vp, d_Vs, nz_, nx_, ny_);
  aveBycInit<<<blocks, threads>>>(d_Rho, d_Byc, nz_, nx_, ny_);

  CHECK(cudaMemcpy(h_Vp, d_Vp, nShape * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_Vs, d_Vs, nShape * sizeof(float), cudaMemcpyDeviceToHost));
}

Model::~Model() {
  free(h_Vp);
  free(h_Vs);
  free(h_Rho);
  free(h_Lambda);
  free(h_Mu);
  free(h_RhoGrad);
  free(h_LambdaGrad);
  free(h_MuGrad);
  CHECK(cudaFree(d_Vp));
  CHECK(cudaFree(d_Vs));
  CHECK(cudaFree(d_Rho));
  CHECK(cudaFree(d_Lambda));
  CHECK(cudaFree(d_Mu));
  CHECK(cudaFree(d_Byc));
  CHECK(cudaFree(d_LambdaGrad));
  CHECK(cudaFree(d_MuGrad));
  CHECK(cudaFree(d_RhoGrad));
}