#include "Boundary.h"
#include "Parameter.h"
#include "utilities.h"

Bnd::Bnd(const Parameter &para) {
  withAdj_ = para.withAdj();
  if (withAdj_) {
    nz_ = para.nz();
    nx_ = para.nx();
    ny_ = para.ny();
    nPml_ = para.nPoints_pml();
    nSteps_ = para.nSteps();

    nzBnd_ = nz_ - 2 * nPml_ + 4;
    nxBnd_ = nx_ - 2 * nPml_ + 4;
    nyBnd_ = ny_ - 2 * nPml_ + 4;
    nLayerStore_ = 5;

    len_Bnd_vec_ =
        2 * (nLayerStore_ * (nzBnd_ * nxBnd_) + nLayerStore_ * (nxBnd_ * nyBnd_) + nLayerStore_ * (nyBnd_ * nzBnd_));  // store n layers

    // allocate the boundary vector in the device
    CHECK(cudaMalloc((void **)&d_Bnd_szz,
                     len_Bnd_vec_ * nSteps_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Bnd_sxz,
                     len_Bnd_vec_ * nSteps_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Bnd_sxx,
                     len_Bnd_vec_ * nSteps_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Bnd_syy,
                     len_Bnd_vec_ * nSteps_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Bnd_sxy,
                     len_Bnd_vec_ * nSteps_ * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Bnd_syz,
                     len_Bnd_vec_ * nSteps_ * sizeof(float)));

    CHECK(
        cudaMalloc((void **)&d_Bnd_vz, len_Bnd_vec_ * nSteps_ * sizeof(float)));
    CHECK(
        cudaMalloc((void **)&d_Bnd_vx, len_Bnd_vec_ * nSteps_ * sizeof(float)));
    CHECK(
        cudaMalloc((void **)&d_Bnd_vy, len_Bnd_vec_ * nSteps_ * sizeof(float)));
  }
}

Bnd::~Bnd() {
  if (withAdj_) {
    CHECK(cudaFree(d_Bnd_szz));
    CHECK(cudaFree(d_Bnd_sxz));
    CHECK(cudaFree(d_Bnd_sxx));
    CHECK(cudaFree(d_Bnd_syy));
    CHECK(cudaFree(d_Bnd_sxy));
    CHECK(cudaFree(d_Bnd_syz));
    CHECK(cudaFree(d_Bnd_vz));
    CHECK(cudaFree(d_Bnd_vx));
    CHECK(cudaFree(d_Bnd_vy));
  }
}

void Bnd::field_from_bnd(float *d_szz, float *d_sxz, float *d_sxx, float *d_syy,
                         float *d_sxy, float *d_syz, float *d_vz, float *d_vx, float *d_vy, int indT) {
  from_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_szz, d_Bnd_szz, nz_, nx_, ny_, nzBnd_,
                                             nxBnd_, nyBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nSteps_);

  from_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_sxz, d_Bnd_sxz, nz_, nx_, ny_, nzBnd_,
                                             nxBnd_, nyBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nSteps_);

  from_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_sxx, d_Bnd_sxx, nz_, nx_, ny_, nzBnd_,
                                             nxBnd_, nyBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nSteps_);

  from_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_syy, d_Bnd_syy, nz_, nx_, ny_, nzBnd_,
                                             nxBnd_, nyBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nSteps_);

  from_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_sxy, d_Bnd_sxy, nz_, nx_, ny_, nzBnd_,
                                             nxBnd_, nyBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nSteps_);

  from_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_syz, d_Bnd_syz, nz_, nx_, ny_, nzBnd_,
                                             nxBnd_, nyBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nSteps_);

  from_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_vz, d_Bnd_vz, nz_, nx_, ny_, nzBnd_,
                                             nxBnd_, nyBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nSteps_);

  from_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_vx, d_Bnd_vx, nz_, nx_, ny_, nzBnd_,
                                             nxBnd_, nyBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nSteps_);

  from_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_vy, d_Bnd_vy, nz_, nx_, ny_, nzBnd_,
                                             nxBnd_, nyBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nSteps_);
}

void Bnd::field_to_bnd(float *d_szz, float *d_sxz, float *d_sxx,
                       float *d_syy, float *d_sxy, float *d_syz,
                       float *d_vz, float *d_vx, float *d_vy, int indT, bool if_stress) {
  if (if_stress) {
    to_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_szz, d_Bnd_szz, nz_, nx_, ny_, nzBnd_,
                                             nxBnd_, nyBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nSteps_);

    to_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_sxz, d_Bnd_sxz, nz_, nx_, ny_, nzBnd_,
                                             nxBnd_, nyBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nSteps_);

    to_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_sxx, d_Bnd_sxx, nz_, nx_, ny_, nzBnd_,
                                             nxBnd_, nyBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nSteps_);

    to_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_syy, d_Bnd_syy, nz_, nx_, ny_, nzBnd_,
                                             nxBnd_, nyBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nSteps_);

    to_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_sxy, d_Bnd_sxy, nz_, nx_, ny_, nzBnd_,
                                             nxBnd_, nyBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nSteps_);

    to_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_syz, d_Bnd_syz, nz_, nx_, ny_, nzBnd_,
                                             nxBnd_, nyBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nSteps_);

  } else {
    to_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_vz, d_Bnd_vz, nz_, nx_, ny_, nzBnd_,
                                             nxBnd_, nyBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nSteps_);

    to_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_vx, d_Bnd_vx, nz_, nx_, ny_, nzBnd_,
                                             nxBnd_, nyBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nSteps_);

    to_bnd<<<(len_Bnd_vec_ + 31) / 32, 32>>>(d_vy, d_Bnd_vy, nz_, nx_, ny_, nzBnd_,
                                             nxBnd_, nyBnd_, len_Bnd_vec_, nLayerStore_,
                                             indT, nPml_, nSteps_);
  }
}