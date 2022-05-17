// Dongzhuo Li 05/06/2018
#include <chrono>
#include <string>
#include <iostream>
#include "Boundary.h"
#include "Cpml.h"
#include "Model.h"
#include "Parameter.h"
#include "Src_Rec.h"
#include "utilities.h"
using std::string;

#define VERBOSE
// #define DEBUG
/*
        double misfit
        double *grad_Lambda : gradients of Lambda (lame parameter)
        double *grad_Mu : gradients of Mu (shear modulus)
        double *grad_Rho : gradients of density
        double *grad_stf : gradients of source time function
        double *Lambda : lame parameter (Mega Pascal)
        double *Mu : shear modulus (Mega Pascal)
        double *Rho : density
        double *stf : source time function of all shots
        int calc_id :
                                        calc_id = 0  -- compute residual
                                        calc_id = 1  -- compute gradient
                                        calc_id = 2  -- compute observation only
        int gpu_id  :   CUDA_VISIBLE_DEVICES
        int group_size: number of shots in the group
        int *shot_ids :   processing shot shot_ids
        string para_fname :  parameter path
        // string survey_fname :  survey file (src/rec) path
        // string data_dir : data directory
        // string scratch_dir : temporary files
*/
extern "C" void cufd(float *misfit, float *grad_Lambda, float *grad_Mu,
          float *grad_Rho, const float *Lambda,
          const float *Mu, const float *Rho, const float *stf, int calc_id,
          const int gpu_id, const int group_size, const int *shot_ids,
          const string para_fname) {

  CHECK(cudaSetDevice(gpu_id));

  auto start0 = std::chrono::high_resolution_clock::now();

  if (calc_id < 0 || calc_id > 2) {
    printf("Invalid calc_id %d\n", calc_id);
    exit(0);
  }

  Parameter para(para_fname, calc_id);
  int nz = para.nz();
  int nx = para.nx();
  int ny = para.ny();
  int nShape = nz * nx * ny;
  int nPml = para.nPoints_pml();
  float dz = para.dz();
  float dx = para.dx();
  float dy = para.dy();
  float dt = para.dt();
  float f0 = para.f0();
  int nrec = 1;
  float win_ratio = 0.005;
  int nSteps = para.nSteps();

  // transpose models and convert to float
  float *fLambda, *fMu, *fRho;
  fLambda = (float *)malloc(nShape * sizeof(float));
  fMu = (float *)malloc(nShape * sizeof(float));
  fRho = (float *)malloc(nShape * sizeof(float));
  for (int i = 0; i < nz; i++) {
    for (int j = 0; j < nx; j++) {
      for (int k = 0; k < ny; k++) {
        fLambda[k * (nx*nz) + j*nz + i] = Lambda[i * (nx*ny) + j * (ny) + k];
        fMu[k * (nx*nz) + j*nz + i] = Mu[i * (nx*ny) + j * (ny) + k];
        fRho[k * (nx*nz) + j*nz + i] = Rho[i * (nx*ny) + j * (ny) + k];
      }
    }
  }// The new dimension of model is (nx, ny, nz)!
  // initial the model
  Model model(para, fLambda, fMu, fRho);
  // calculate the cpml bound to absorb the reflection of boundary.
  Cpml cpml(para, model);
  // For recovery the forward wavefields.
  Bnd boundaries(para);
  // initial the source and reveive condition.
  Src_Rec src_rec(para, para.survey_fname(), stf, group_size, shot_ids);

  // compute Courant number
  compCourantNumber(model.h_Vp, nShape, dt, dz, dx, dy);

  const dim3 threads(32, 1, 32);
  const int gridx = (nx + threads.x - 1) / threads.x;
  const int gridy = (ny + threads.y - 1) / threads.y;
  const int gridz = (nz + threads.z - 1) / threads.z;
  const dim3 blocks(gridx, gridy, gridz);
  // printf("gridx = %d\n", gridx);
  // printf("gridy = %d\n", gridy);
  // printf("gridz = %d\n", gridz);
  // printf("nz = %d\n", nz);
  // printf("ny = %d\n", ny);
  // printf("nx = %d\n", nx);
  
  float *d_vz, *d_vx, *d_vy, *d_szz, *d_sxx, *d_sxz, *d_syy, *d_sxy, *d_syz,
        *d_vz_adj, *d_vx_adj, *d_vy_adj, *d_szz_adj, *d_sxx_adj, *d_sxz_adj, *d_syy_adj, *d_sxy_adj, *d_syz_adj;
  float *d_mem_dvz_dz, *d_mem_dvz_dx, *d_mem_dvz_dy, *d_mem_dvx_dz, *d_mem_dvx_dx, *d_mem_dvx_dy,
        *d_mem_dvy_dz, *d_mem_dvy_dx, *d_mem_dvy_dy;
  float *d_mem_dszz_dz, *d_mem_dsxx_dx, *d_mem_dsyy_dy,
        *d_mem_dsxz_dz, *d_mem_dsxz_dx, *d_mem_dsxy_dx, *d_mem_dsxy_dy, *d_mem_dsyz_dy, *d_mem_dsyz_dz;
  float *d_l2Obj_temp_x = (float *)malloc(sizeof(float));
  float *d_l2Obj_temp_y = (float *)malloc(sizeof(float));
  float *d_l2Obj_temp_z = (float *)malloc(sizeof(float));
  float *h_l2Obj_temp_x = (float *)malloc(sizeof(float));
  float *h_l2Obj_temp_y = (float *)malloc(sizeof(float));
  float *h_l2Obj_temp_z = (float *)malloc(sizeof(float));
  float h_l2Obj = 0.0;
  float *d_gauss_amp;
  float *d_data_x;
  float *d_data_y;
  float *d_data_z;
  float *d_data_obs_x;
  float *d_data_obs_y;
  float *d_data_obs_z;
  float *d_res_x;
  float *d_res_y;
  float *d_res_z;

  // ceshi
  // float *h_ceshi1, *h_ceshi2, *d_ceshi;
  // h_ceshi1 = (float *)malloc(nShape * sizeof(float));
  // h_ceshi2 = (float *)malloc(nShape * sizeof(float));
  // CHECK(cudaMalloc((void **)&d_ceshi, nShape * sizeof(float)));

  // initialArray(h_ceshi1, nShape, 1.0);
  // CHECK(cudaMemcpyAsync(d_ceshi, h_ceshi1, nShape * sizeof(float), cudaMemcpyHostToDevice));
  // auto start_ceshi = std::chrono::high_resolution_clock::now();
  // for (int i = 0; i<100000; i++){
  //   initialArrayGPU<<<blocks, threads>>>(d_ceshi, nz, nx, ny, -1.0);
  // }
  
  // auto finish_ceshi = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed_ceshi = finish_ceshi - start_ceshi;
  // std::cout << "time: " << elapsed_ceshi.count() << " second(s)"  << std::endl;

  // CHECK(cudaMemcpy(h_ceshi2, d_ceshi, nShape * sizeof(float), cudaMemcpyDeviceToHost));
  // fileBinWrite(h_ceshi2, nShape, para.data_dir_name() + "/ceshi.bin");
  // cudaDeviceSynchronize();
  // cudaError_t err = cudaGetLastError();

  // exit(0);
  // over

  CHECK(cudaMalloc((void **)&d_vz, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_vx, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_vy, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_szz, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_sxx, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_sxz, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_syy, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_sxy, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_syz, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_vz_adj, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_vx_adj, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_vy_adj, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_szz_adj, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_sxx_adj, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_sxz_adj, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_syy_adj, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_sxy_adj, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_syz_adj, nShape * sizeof(float)));

  CHECK(cudaMalloc((void **)&d_mem_dvz_dz, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dvz_dx, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dvz_dy, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dvx_dz, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dvx_dx, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dvx_dy, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dvy_dz, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dvy_dx, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dvy_dy, nShape * sizeof(float)));

  CHECK(cudaMalloc((void **)&d_mem_dszz_dz, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dsxx_dx, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dsyy_dy, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dsxz_dz, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dsxz_dx, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dsxy_dx, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dsxy_dy, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dsyz_dy, nShape * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_mem_dsyz_dz, nShape * sizeof(float)));

  CHECK(cudaMalloc((void **)&d_l2Obj_temp_x, 1 * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_l2Obj_temp_y, 1 * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_l2Obj_temp_z, 1 * sizeof(float)));
  CHECK(cudaMalloc((void **)&d_gauss_amp, 729 * sizeof(float)));

  //   float *h_snap, *h_snap_back, *h_snap_adj;
  // h_snap = (float *)malloc(nShape * sizeof(float));
  // h_snap_back = (float *)malloc(nShape * sizeof(float));
  // h_snap_adj = (float *)malloc(nShape * sizeof(float));

  dim3 blocks3(1, 1, 9);
  dim3 threads3(32, 32, 1);
  src_rec_gauss_amp<<<blocks3, threads3>>>(d_gauss_amp, 9, 9, 9);

  cudaStream_t *streams = (cudaStream_t *)malloc(group_size * sizeof(cudaStream_t));


  #ifdef VERBOSE  
  auto finish0 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed0 = finish0 - start0;
      std::cout << "Initialization time: " << elapsed0.count() << " second(s)"
            << std::endl;
     auto start = std::chrono::high_resolution_clock::now();          
  #endif


  // NOTE Processing Shot
  for (int iShot = 0; iShot < group_size; iShot++) {
// #ifdef VERBOSE
    printf("	Processing shot %d\n", shot_ids[iShot]);
// #endif
    CHECK(cudaStreamCreate(&streams[iShot]));

    initialArrayGPU<<<blocks, threads>>>(d_vz, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_vx, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_vy, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_vz_adj, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_vx_adj, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_vy_adj, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_szz, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_sxx, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_sxz, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_syy, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_sxy, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_syz, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_szz_adj, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_sxx_adj, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_sxz_adj, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_syy_adj, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_sxy_adj, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_syz_adj, nz, nx, ny, 0.0);

    initialArrayGPU<<<blocks, threads>>>(d_mem_dvz_dz, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_mem_dvz_dx, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_mem_dvz_dy, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_mem_dvx_dz, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_mem_dvx_dx, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_mem_dvx_dy, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_mem_dvy_dz, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_mem_dvy_dx, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_mem_dvy_dy, nz, nx, ny, 0.0);

    initialArrayGPU<<<blocks, threads>>>(d_mem_dszz_dz, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_mem_dsxx_dx, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_mem_dsyy_dy, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_mem_dsxz_dz, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_mem_dsxz_dx, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_mem_dsxy_dx, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_mem_dsxy_dy, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_mem_dsyz_dy, nz, nx, ny, 0.0);
    initialArrayGPU<<<blocks, threads>>>(d_mem_dsyz_dz, nz, nx, ny, 0.0);

    nrec = src_rec.vec_nrec.at(iShot);

    dim3 threads_t(32, 32);
    dim3 blocks_t((nSteps + 32 - 1) / 32, (nrec + 32 - 1) / 32);
    CHECK(cudaMalloc((void **)&d_data_x, nrec * nSteps * sizeof(float)));
    initial2DArrayGPU<<<blocks_t, threads_t>>>(d_data_x, nSteps, nrec, 0.0);
    CHECK(cudaMalloc((void **)&d_data_y, nrec * nSteps * sizeof(float)));
    initial2DArrayGPU<<<blocks_t, threads_t>>>(d_data_y, nSteps, nrec, 0.0);
    CHECK(cudaMalloc((void **)&d_data_z, nrec * nSteps * sizeof(float)));
    initial2DArrayGPU<<<blocks_t, threads_t>>>(d_data_z, nSteps, nrec, 0.0);

    if (para.if_res()) {
      fileBinLoad(src_rec.vec_data_obs_x.at(iShot), nSteps * nrec,
                  para.data_dir_name() + "/Shot_x" +
                      std::to_string(shot_ids[iShot]) + ".bin");
      CHECK(cudaMalloc((void **)&d_data_obs_x, nrec * nSteps * sizeof(float)));
      CHECK(cudaMalloc((void **)&d_res_x, nrec * nSteps * sizeof(float)));
      initial2DArrayGPU<<<blocks_t, threads_t>>>(d_data_obs_x, nSteps, nrec, 0.0);
      initial2DArrayGPU<<<blocks_t, threads_t>>>(d_res_x, nSteps, nrec, 0.0);
      CHECK(cudaMemcpyAsync(d_data_obs_x, src_rec.vec_data_obs_x.at(iShot),
                            nrec * nSteps * sizeof(float),
                            cudaMemcpyHostToDevice, streams[iShot]));

      fileBinLoad(src_rec.vec_data_obs_y.at(iShot), nSteps * nrec,
                  para.data_dir_name() + "/Shot_y" +
                      std::to_string(shot_ids[iShot]) + ".bin");
      CHECK(cudaMalloc((void **)&d_data_obs_y, nrec * nSteps * sizeof(float)));
      CHECK(cudaMalloc((void **)&d_res_y, nrec * nSteps * sizeof(float)));
      initial2DArrayGPU<<<blocks_t, threads_t>>>(d_data_obs_y, nSteps, nrec, 0.0);
      initial2DArrayGPU<<<blocks_t, threads_t>>>(d_res_y, nSteps, nrec, 0.0);
      CHECK(cudaMemcpyAsync(d_data_obs_y, src_rec.vec_data_obs_y.at(iShot),
                            nrec * nSteps * sizeof(float),
                            cudaMemcpyHostToDevice, streams[iShot]));

      fileBinLoad(src_rec.vec_data_obs_z.at(iShot), nSteps * nrec,
                  para.data_dir_name() + "/Shot_z" +
                      std::to_string(shot_ids[iShot]) + ".bin");
      CHECK(cudaMalloc((void **)&d_data_obs_z, nrec * nSteps * sizeof(float)));
      CHECK(cudaMalloc((void **)&d_res_z, nrec * nSteps * sizeof(float)));
      initial2DArrayGPU<<<blocks_t, threads_t>>>(d_data_obs_z, nSteps, nrec, 0.0);
      initial2DArrayGPU<<<blocks_t, threads_t>>>(d_res_z, nSteps, nrec, 0.0);
      CHECK(cudaMemcpyAsync(d_data_obs_z, src_rec.vec_data_obs_z.at(iShot),
                            nrec * nSteps * sizeof(float),
                            cudaMemcpyHostToDevice, streams[iShot]));
    }

    // ------------------------ time loop ----------------------------
    for (int it = 0; it <= nSteps - 2; it++) {
      // ================= 3Delastic =====================
      if (para.withAdj()) {
        // save and record from the beginning
        boundaries.field_from_bnd(d_szz, d_sxz, d_sxx, d_syy, d_sxy, d_syz, d_vz, d_vx, d_vy, it);
      }

      stress<<<blocks, threads>>>(
          d_vz, d_vx, d_vy, d_szz, d_sxx, d_sxz, d_syy, d_sxy, d_syz, d_mem_dvz_dz, d_mem_dvz_dx,
          d_mem_dvz_dy, d_mem_dvx_dz, d_mem_dvx_dx, d_mem_dvx_dy, d_mem_dvy_dz, d_mem_dvy_dx, d_mem_dvy_dy,
          model.d_Lambda, model.d_Mu, model.d_Rho, cpml.d_K_z, cpml.d_a_z, cpml.d_b_z,
          cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x, cpml.d_a_x,
          cpml.d_b_x, cpml.d_K_x_half, cpml.d_a_x_half, cpml.d_b_x_half, cpml.d_K_y, cpml.d_a_y, cpml.d_b_y,
          cpml.d_K_y_half, cpml.d_a_y_half, cpml.d_b_y_half, nz, nx, ny, dt, dz, dx, dy, nPml, true, d_szz_adj,
          d_sxx_adj, d_sxz_adj, d_syy_adj, d_sxy_adj, d_syz_adj, model.d_LambdaGrad, model.d_MuGrad);

      add_source<<<blocks3, threads3>>>(d_szz, d_sxx, d_syy, src_rec.vec_source.at(iShot)[it],
                                        nz, nx, ny, true, src_rec.vec_z_src.at(iShot),
                                        src_rec.vec_x_src.at(iShot), src_rec.vec_y_src.at(iShot), dt, d_gauss_amp);     
      velocity<<<blocks, threads>>>(
          d_vz, d_vx, d_vy, d_szz, d_sxx, d_sxz, d_syy, d_sxy, d_syz, d_mem_dszz_dz, d_mem_dsxz_dx,
          d_mem_dsxz_dz, d_mem_dsxx_dx, d_mem_dsxy_dx, d_mem_dsxy_dy, d_mem_dsyy_dy, d_mem_dsyz_dy,
          d_mem_dsyz_dz, model.d_Lambda, model.d_Mu,
          model.d_Byc, cpml.d_K_z, cpml.d_a_z,
          cpml.d_b_z, cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half,
          cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, cpml.d_K_x_half, cpml.d_a_x_half,
          cpml.d_b_x_half, cpml.d_K_y, cpml.d_a_y, cpml.d_b_y, cpml.d_K_y_half, cpml.d_a_y_half,
          cpml.d_b_y_half, nz, nx, ny, dt, dz, dx, dy, nPml, true, d_vz_adj,
          d_vx_adj, d_vy_adj, model.d_RhoGrad);


      // if (iShot == 0) {
      //   // printf("it = %d\n", it);
      //     CHECK(cudaMemcpy(h_snap, d_vz, nShape * sizeof(float),
      //                     cudaMemcpyDeviceToHost));
      //     fileBinWrite(h_snap, nShape, para.data_dir_name() + 
      //                 "/SnapGPU_zy" + std::to_string(it) + ".bin");
      //   }

      recording<<<(nrec + 31) / 32, 32>>>(
          d_vz, d_vx, d_vy, nz, nx, d_data_x, d_data_y, d_data_z, iShot, it + 1, nSteps, nrec,
          src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot), src_rec.d_vec_y_rec.at(iShot));
    }

    if (!para.if_res()) {
    CHECK(cudaMemcpyAsync(src_rec.vec_data_x.at(iShot), d_data_x,
                           nSteps * nrec * sizeof(float),
                           cudaMemcpyDeviceToHost, streams[iShot]));
                           
    CHECK(cudaMemcpyAsync(src_rec.vec_data_y.at(iShot), d_data_y,
                           nSteps * nrec * sizeof(float),
                           cudaMemcpyDeviceToHost, streams[iShot]));

    CHECK(cudaMemcpyAsync(src_rec.vec_data_z.at(iShot), d_data_z,
                           nSteps * nrec * sizeof(float),
                           cudaMemcpyDeviceToHost, streams[iShot]));
    }

    // compute residuals
    if (para.if_res()) {
      if (para.if_win()) {
        cuda_window<<<blocks_t, threads_t>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            src_rec.vec_srcweights.at(iShot), win_ratio, d_data_obs_x);
        cuda_window<<<blocks_t, threads_t>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            src_rec.vec_srcweights.at(iShot), win_ratio, d_data_x);

        cuda_window<<<blocks_t, threads_t>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            src_rec.vec_srcweights.at(iShot), win_ratio, d_data_obs_y);
        cuda_window<<<blocks_t, threads_t>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            src_rec.vec_srcweights.at(iShot), win_ratio, d_data_y);

        cuda_window<<<blocks_t, threads_t>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            src_rec.vec_srcweights.at(iShot), win_ratio, d_data_obs_z);
        cuda_window<<<blocks_t, threads_t>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            src_rec.vec_srcweights.at(iShot), win_ratio, d_data_z);
      } else {
        cuda_window<<<blocks_t, threads_t>>>(nSteps, nrec, dt, win_ratio, d_data_obs_x);
        cuda_window<<<blocks_t, threads_t>>>(nSteps, nrec, dt, win_ratio, d_data_obs_y);
        cuda_window<<<blocks_t, threads_t>>>(nSteps, nrec, dt, win_ratio, d_data_obs_z);
        cuda_window<<<blocks_t, threads_t>>>(nSteps, nrec, dt, win_ratio, d_data_x);
        cuda_window<<<blocks_t, threads_t>>>(nSteps, nrec, dt, win_ratio, d_data_y);
        cuda_window<<<blocks_t, threads_t>>>(nSteps, nrec, dt, win_ratio, d_data_z);
      }

      // objective function
      gpuMinus<<<blocks_t, threads_t>>>(d_res_x, d_data_obs_x, d_data_x, nSteps, nrec);
      
      gpuMinus<<<blocks_t, threads_t>>>(d_res_y, d_data_obs_y, d_data_y, nSteps, nrec);
      gpuMinus<<<blocks_t, threads_t>>>(d_res_z, d_data_obs_z, d_data_z, nSteps, nrec);
      cuda_cal_objective<<<1, 512>>>(d_l2Obj_temp_x, d_res_x, nSteps * nrec);
      cuda_cal_objective<<<1, 512>>>(d_l2Obj_temp_y, d_res_y, nSteps * nrec);
      cuda_cal_objective<<<1, 512>>>(d_l2Obj_temp_z, d_res_z, nSteps * nrec);

      CHECK(cudaMemcpy(h_l2Obj_temp_x, d_l2Obj_temp_x, sizeof(float), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(h_l2Obj_temp_y, d_l2Obj_temp_y, sizeof(float), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(h_l2Obj_temp_z, d_l2Obj_temp_z, sizeof(float), cudaMemcpyDeviceToHost));
      h_l2Obj += h_l2Obj_temp_x[0] + h_l2Obj_temp_y[0] + h_l2Obj_temp_z[0];

      // windowing again (adjoint)
      if (para.if_win()) {
        cuda_window<<<blocks_t, threads_t>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            src_rec.vec_srcweights.at(iShot), win_ratio, d_res_x);
        cuda_window<<<blocks_t, threads_t>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            src_rec.vec_srcweights.at(iShot), win_ratio, d_res_y);
        cuda_window<<<blocks_t, threads_t>>>(
            nSteps, nrec, dt, src_rec.d_vec_win_start.at(iShot),
            src_rec.d_vec_win_end.at(iShot), src_rec.d_vec_weights.at(iShot),
            src_rec.vec_srcweights.at(iShot), win_ratio, d_res_z);
      } else {
        cuda_window<<<blocks_t, threads_t>>>(nSteps, nrec, dt, win_ratio, d_res_x);
        cuda_window<<<blocks_t, threads_t>>>(nSteps, nrec, dt, win_ratio, d_res_y);
        cuda_window<<<blocks_t, threads_t>>>(nSteps, nrec, dt, win_ratio, d_res_z);
      }

      CHECK(cudaMemcpyAsync(src_rec.vec_res_x.at(iShot), d_res_x, nSteps * nrec * sizeof(float), 
                            cudaMemcpyDeviceToHost, streams[iShot]));  // test
      CHECK(cudaMemcpyAsync(src_rec.vec_data_x.at(iShot), d_data_x, nSteps * nrec * sizeof(float), 
                            cudaMemcpyDeviceToHost, streams[iShot]));  // test
      CHECK(cudaMemcpyAsync(src_rec.vec_data_obs_x.at(iShot), d_data_obs_x, nSteps * nrec * sizeof(float), 
                            cudaMemcpyDeviceToHost, streams[iShot]));  // save preconditioned observed
      
      CHECK(cudaMemcpyAsync(src_rec.vec_res_y.at(iShot), d_res_y, nSteps * nrec * sizeof(float), 
                            cudaMemcpyDeviceToHost, streams[iShot]));  // test
      CHECK(cudaMemcpyAsync(src_rec.vec_data_y.at(iShot), d_data_y, nSteps * nrec * sizeof(float), 
                            cudaMemcpyDeviceToHost, streams[iShot]));  // test
      CHECK(cudaMemcpyAsync(src_rec.vec_data_obs_y.at(iShot), d_data_obs_y, nSteps * nrec * sizeof(float), 
                            cudaMemcpyDeviceToHost, streams[iShot]));  // save preconditioned observed
      
      CHECK(cudaMemcpyAsync(src_rec.vec_res_z.at(iShot), d_res_z, nSteps * nrec * sizeof(float), 
                            cudaMemcpyDeviceToHost, streams[iShot]));  // test
      CHECK(cudaMemcpyAsync(src_rec.vec_data_z.at(iShot), d_data_z, nSteps * nrec * sizeof(float), 
                            cudaMemcpyDeviceToHost, streams[iShot]));  // test
      CHECK(cudaMemcpyAsync(src_rec.vec_data_obs_z.at(iShot), d_data_obs_z, nSteps * nrec * sizeof(float), 
                            cudaMemcpyDeviceToHost, streams[iShot]));  // save preconditioned observed
                          
      CHECK(cudaMemcpy(src_rec.vec_source.at(iShot), src_rec.d_vec_source.at(iShot), nSteps * sizeof(float),
                            cudaMemcpyDeviceToHost));
    }
    // =================
    cudaDeviceSynchronize();

    if (para.withAdj()) {
      // --------------------- Backward ----------------------------
      // initialization
      initialArrayGPU<<<blocks, threads>>>(d_vz_adj, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_vx_adj, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_vy_adj, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_szz_adj, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_sxx_adj, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_sxz_adj, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_syy_adj, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_sxy_adj, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_syz_adj, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_mem_dvz_dz, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_mem_dvz_dx, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_mem_dvz_dy, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_mem_dvx_dz, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_mem_dvx_dx, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_mem_dvx_dy, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_mem_dvy_dy, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_mem_dvy_dz, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_mem_dvy_dx, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_mem_dszz_dz, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_mem_dsxz_dx, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_mem_dsxz_dz, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_mem_dsxx_dx, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_mem_dsxy_dx, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_mem_dsxy_dy, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_mem_dsyy_dy, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_mem_dsyz_dy, nz, nx, ny, 0.0);
      initialArrayGPU<<<blocks, threads>>>(d_mem_dsyz_dz, nz, nx, ny, 0.0);

      velocity_adj<<<blocks, threads>>>(
          d_vz_adj, d_vx_adj, d_vy_adj, d_szz_adj, d_sxx_adj, 
          d_sxz_adj, d_syy_adj, d_sxy_adj, d_syz_adj, 
          d_mem_dszz_dz, d_mem_dsxz_dx, d_mem_dsxz_dz, d_mem_dsxx_dx, 
          d_mem_dsxy_dx, d_mem_dsxy_dy, d_mem_dsyy_dy, d_mem_dsyz_dy, d_mem_dsyz_dz, 
          d_mem_dvz_dz, d_mem_dvz_dx, d_mem_dvz_dy, d_mem_dvx_dz, d_mem_dvx_dx, 
          d_mem_dvx_dy, d_mem_dvy_dz, d_mem_dvy_dx, d_mem_dvy_dy,
          model.d_Lambda, model.d_Mu, model.d_Rho, 
          model.d_Byc, cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x_half,
          cpml.d_a_x_half, cpml.d_b_x_half, cpml.d_K_y_half,
          cpml.d_a_y_half, cpml.d_b_y_half, cpml.d_K_z, cpml.d_a_z, cpml.d_b_z,
          cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, cpml.d_K_y, cpml.d_a_y, cpml.d_b_y, 
          nz, nx, ny, dt, dz, dx, dy, nPml);

      res_injection<<<(nrec + 31) / 32, 32>>>(
          d_vx_adj, d_vy_adj, d_vz_adj, nz, nx, d_res_x, d_res_y, d_res_z, nSteps - 1, dt, nSteps, nrec,
          src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot), src_rec.d_vec_y_rec.at(iShot));

      stress_adj<<<blocks, threads>>>(
          d_vz_adj, d_vx_adj, d_vy_adj, d_szz_adj, d_sxx_adj, 
          d_sxz_adj, d_syy_adj, d_sxy_adj, d_syz_adj, 
          d_mem_dszz_dz, d_mem_dsxz_dx, d_mem_dsxz_dz, d_mem_dsxx_dx, 
          d_mem_dsxy_dx, d_mem_dsxy_dy, d_mem_dsyy_dy, d_mem_dsyz_dy, d_mem_dsyz_dz, 
          d_mem_dvz_dz, d_mem_dvz_dx, d_mem_dvz_dy, d_mem_dvx_dz, d_mem_dvx_dx, 
          d_mem_dvx_dy, d_mem_dvy_dz, d_mem_dvy_dx, d_mem_dvy_dy,
          model.d_Lambda, model.d_Mu, model.d_Rho, 
          model.d_Byc, cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x_half,
          cpml.d_a_x_half, cpml.d_b_x_half, cpml.d_K_y_half,
          cpml.d_a_y_half, cpml.d_b_y_half, cpml.d_K_z, cpml.d_a_z, cpml.d_b_z,
          cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, cpml.d_K_y, cpml.d_a_y, cpml.d_b_y, 
          nz, nx, ny, dt, dz, dx, dy, nPml);

      for (int it = nSteps - 2; it >= 0; it--) {

        velocity<<<blocks, threads>>>(
          d_vz, d_vx, d_vy, d_szz, d_sxx, d_sxz, d_syy, d_sxy, d_syz, d_mem_dszz_dz, d_mem_dsxz_dx,
          d_mem_dsxz_dz, d_mem_dsxx_dx, d_mem_dsxy_dx, d_mem_dsxy_dy, d_mem_dsyy_dy, d_mem_dsyz_dy,
          d_mem_dsyz_dz, model.d_Lambda, model.d_Mu,
          model.d_Byc, cpml.d_K_z, cpml.d_a_z,
          cpml.d_b_z, cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half,
          cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, cpml.d_K_x_half, cpml.d_a_x_half,
          cpml.d_b_x_half, cpml.d_K_y, cpml.d_a_y, cpml.d_b_y, cpml.d_K_y_half, cpml.d_a_y_half,
          cpml.d_b_y_half, nz, nx, ny, dt, dz, dx, dy, nPml, false, d_vz_adj,
          d_vx_adj, d_vy_adj, model.d_RhoGrad);

        boundaries.field_to_bnd(d_szz, d_sxz, d_sxx, d_syy, d_sxy, d_syz, d_vz, d_vx, d_vy, it, false);

        add_source<<<blocks3, threads3>>>(
            d_szz, d_sxx, d_syy, src_rec.vec_source.at(iShot)[it], nz, nx, ny, false,
            src_rec.vec_z_src.at(iShot), src_rec.vec_x_src.at(iShot), src_rec.vec_y_src.at(iShot), dt, d_gauss_amp);

        stress<<<blocks, threads>>>(
          d_vz, d_vx, d_vy, d_szz, d_sxx, d_sxz, d_syy, d_sxy, d_syz, d_mem_dvz_dz, d_mem_dvz_dx,
          d_mem_dvz_dy, d_mem_dvx_dz, d_mem_dvx_dx, d_mem_dvx_dy, d_mem_dvy_dz, d_mem_dvy_dx, d_mem_dvy_dy,
          model.d_Lambda, model.d_Mu, model.d_Rho, cpml.d_K_z, cpml.d_a_z, cpml.d_b_z,
          cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x, cpml.d_a_x,
          cpml.d_b_x, cpml.d_K_x_half, cpml.d_a_x_half, cpml.d_b_x_half, cpml.d_K_y, cpml.d_a_y, cpml.d_b_y,
          cpml.d_K_y_half, cpml.d_a_y_half, cpml.d_b_y_half, nz, nx, ny, dt, dz, dx, dy, nPml, false, d_szz_adj,
          d_sxx_adj, d_sxz_adj, d_syy_adj, d_sxy_adj, d_syz_adj, model.d_LambdaGrad, model.d_MuGrad);

        boundaries.field_to_bnd(d_szz, d_sxz, d_sxx, d_syy, d_sxy, d_syz, d_vz, d_vx, d_vy, it, true);

        velocity_adj<<<blocks, threads>>>(
          d_vz_adj, d_vx_adj, d_vy_adj, d_szz_adj, d_sxx_adj, 
          d_sxz_adj, d_syy_adj, d_sxy_adj, d_syz_adj, 
          d_mem_dszz_dz, d_mem_dsxz_dx, d_mem_dsxz_dz, d_mem_dsxx_dx, 
          d_mem_dsxy_dx, d_mem_dsxy_dy, d_mem_dsyy_dy, d_mem_dsyz_dy, d_mem_dsyz_dz, 
          d_mem_dvz_dz, d_mem_dvz_dx, d_mem_dvz_dy, d_mem_dvx_dz, d_mem_dvx_dx, 
          d_mem_dvx_dy, d_mem_dvy_dz, d_mem_dvy_dx, d_mem_dvy_dy,
          model.d_Lambda, model.d_Mu, model.d_Rho, 
          model.d_Byc,
          cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x_half,
          cpml.d_a_x_half, cpml.d_b_x_half, cpml.d_K_y_half,
          cpml.d_a_y_half, cpml.d_b_y_half, cpml.d_K_z, cpml.d_a_z, cpml.d_b_z,
          cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, cpml.d_K_y, cpml.d_a_y, cpml.d_b_y, 
          nz, nx, ny, dt, dz, dx, dy, nPml);
        
        res_injection<<<(nrec + 31) / 32, 32>>>(
            d_vx_adj, d_vy_adj, d_vz_adj, nz, nx, d_res_x, d_res_y, d_res_z, it, dt, nSteps, nrec,
            src_rec.d_vec_z_rec.at(iShot), src_rec.d_vec_x_rec.at(iShot), src_rec.d_vec_y_rec.at(iShot));
                           
        stress_adj<<<blocks, threads>>>(
          d_vz_adj, d_vx_adj, d_vy_adj, d_szz_adj, d_sxx_adj, 
          d_sxz_adj, d_syy_adj, d_sxy_adj, d_syz_adj, 
          d_mem_dszz_dz, d_mem_dsxz_dx, d_mem_dsxz_dz, d_mem_dsxx_dx, 
          d_mem_dsxy_dx, d_mem_dsxy_dy, d_mem_dsyy_dy, d_mem_dsyz_dy, d_mem_dsyz_dz, 
          d_mem_dvz_dz, d_mem_dvz_dx, d_mem_dvz_dy, d_mem_dvx_dz, d_mem_dvx_dx, 
          d_mem_dvx_dy, d_mem_dvy_dz, d_mem_dvy_dx, d_mem_dvy_dy,
          model.d_Lambda, model.d_Mu, model.d_Rho, 
          model.d_Byc,
          cpml.d_K_z_half, cpml.d_a_z_half, cpml.d_b_z_half, cpml.d_K_x_half,
          cpml.d_a_x_half, cpml.d_b_x_half, cpml.d_K_y_half,
          cpml.d_a_y_half, cpml.d_b_y_half, cpml.d_K_z, cpml.d_a_z, cpml.d_b_z,
          cpml.d_K_x, cpml.d_a_x, cpml.d_b_x, cpml.d_K_y, cpml.d_a_y, cpml.d_b_y, 
          nz, nx, ny, dt, dz, dx, dy, nPml);

        // if (it == iSnap1 && iShot == 500) {
        //  CHECK(cudaMemcpy(h_snap_back, d_vz, nShape * sizeof(float),
        //                   cudaMemcpyDeviceToHost));
        //  CHECK(cudaMemcpy(h_snap_adj, d_szz_adj, nShape * sizeof(float),
        //                   cudaMemcpyDeviceToHost));
        // }
        // if (iShot == 0) {
        //   CHECK(cudaMemcpy(h_snap_adj, d_vz_adj, nShape * sizeof(float),
        //                     cudaMemcpyDeviceToHost));
        //   fileBinWrite(h_snap_adj, nShape, para.data_dir_name() +
        //                "/SnapGPU_adj_" + std::to_string(it) + ".bin");
        //   CHECK(cudaMemcpy(h_snap, d_vz, nShape * sizeof(float),
        //                   cudaMemcpyDeviceToHost));
        //   fileBinWrite(h_snap, nShape, para.data_dir_name() + 
        //               "/SnapGPU_" + std::to_string(it) + ".bin");
        // }
      }  // the end of backward time loop   
    }  // end bracket of if adj

   CHECK(cudaFree(d_data_x));
   CHECK(cudaFree(d_data_y));
   CHECK(cudaFree(d_data_z));
    if (para.if_res()) {
      CHECK(cudaFree(d_data_obs_x));
      CHECK(cudaFree(d_data_obs_y));
      CHECK(cudaFree(d_data_obs_z));
      CHECK(cudaFree(d_res_x));
      CHECK(cudaFree(d_res_y));
      CHECK(cudaFree(d_res_z));
    }
  }  // the end of shot loop


#ifdef VERBOSE  
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Elapsed time: " << elapsed.count() << " second(s)."
            << std::endl;
#endif

  if (para.withAdj()) {
    // transfer gradients to cpu
    CHECK(cudaMemcpy(model.h_LambdaGrad, model.d_LambdaGrad,
                     nShape * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(model.h_MuGrad, model.d_MuGrad, nShape * sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(model.h_RhoGrad, model.d_RhoGrad, nShape * sizeof(float),
                     cudaMemcpyDeviceToHost));
    for (int i = 0; i < nz; i++) {
      for (int j = 0; j < nx; j++) {
        for (int k = 0; k < ny; k++){        
        grad_Lambda[i * (nx*ny) + j * (ny) + k] = model.h_LambdaGrad[k * (nx*nz) + j*nz + i];
        grad_Mu[i * (nx*ny) + j * (ny) + k] = model.h_MuGrad[k * (nx*nz) + j*nz + i];
        grad_Rho[i * (nx*ny) + j * (ny) + k] = model.h_RhoGrad[k * (nx*nz) + j*nz + i];
        }
      }
    }
//  #ifdef DEBUG
    // fileBinWrite(model.h_LambdaGrad, nShape, para.data_dir_name() + "/LambdaGradient.bin");
    // fileBinWrite(model.h_MuGrad, nShape, para.data_dir_name() + "/MuGradient.bin");
    // fileBinWrite(model.h_RhoGrad, nShape, para.data_dir_name() + "/RhoGradient.bin");
    // exit(0);
//  #endif

    if (para.if_save_scratch()) {
      for (int iShot = 0; iShot < group_size; iShot++) {
        fileBinWrite(src_rec.vec_res_x.at(iShot),
                     nSteps * src_rec.vec_nrec.at(iShot),
                     para.scratch_dir_name() + "/Residual_Shot_x" +
                         std::to_string(shot_ids[iShot]) + ".bin");
        fileBinWrite(src_rec.vec_data_x.at(iShot),
                     nSteps * src_rec.vec_nrec.at(iShot),
                     para.scratch_dir_name() + "/Syn_Shot_x" +
                         std::to_string(shot_ids[iShot]) + ".bin");
        fileBinWrite(src_rec.vec_data_obs_x.at(iShot),
                     nSteps * src_rec.vec_nrec.at(iShot),
                     para.scratch_dir_name() + "/CondObs_Shot_x" +
                         std::to_string(shot_ids[iShot]) + ".bin");

        fileBinWrite(src_rec.vec_res_y.at(iShot),
                     nSteps * src_rec.vec_nrec.at(iShot),
                     para.scratch_dir_name() + "/Residual_Shot_y" +
                         std::to_string(shot_ids[iShot]) + ".bin");
        fileBinWrite(src_rec.vec_data_y.at(iShot),
                     nSteps * src_rec.vec_nrec.at(iShot),
                     para.scratch_dir_name() + "/Syn_Shot_y" +
                         std::to_string(shot_ids[iShot]) + ".bin");
        fileBinWrite(src_rec.vec_data_obs_y.at(iShot),
                     nSteps * src_rec.vec_nrec.at(iShot),
                     para.scratch_dir_name() + "/CondObs_Shot_y" +
                         std::to_string(shot_ids[iShot]) + ".bin");

        fileBinWrite(src_rec.vec_res_z.at(iShot),
                     nSteps * src_rec.vec_nrec.at(iShot),
                     para.scratch_dir_name() + "/Residual_Shot_z" +
                         std::to_string(shot_ids[iShot]) + ".bin");
        fileBinWrite(src_rec.vec_data_z.at(iShot),
                     nSteps * src_rec.vec_nrec.at(iShot),
                     para.scratch_dir_name() + "/Syn_Shot_z" +
                         std::to_string(shot_ids[iShot]) + ".bin");
        fileBinWrite(src_rec.vec_data_obs_z.at(iShot),
                     nSteps * src_rec.vec_nrec.at(iShot),
                     para.scratch_dir_name() + "/CondObs_Shot_z" +
                         std::to_string(shot_ids[iShot]) + ".bin");
      }
    }
  }

  if (!para.if_res()) {
    #ifdef VERBOSE
      auto startSrc = std::chrono::high_resolution_clock::now();
    #endif
    for (int iShot = 0; iShot < group_size; iShot++) {
      fileBinWrite(src_rec.vec_data_x.at(iShot),
                   nSteps  * src_rec.vec_nrec.at(iShot),
                   para.data_dir_name() + "/Shot_x" +
                       std::to_string(shot_ids[iShot]) + ".bin");
      fileBinWrite(src_rec.vec_data_y.at(iShot),
                   nSteps  * src_rec.vec_nrec.at(iShot),
                   para.data_dir_name() + "/Shot_y" +
                       std::to_string(shot_ids[iShot]) + ".bin");
      fileBinWrite(src_rec.vec_data_z.at(iShot),
                   nSteps  * src_rec.vec_nrec.at(iShot),
                   para.data_dir_name() + "/Shot_z" +
                       std::to_string(shot_ids[iShot]) + ".bin");
      // printf("nSteps = %d\n", nSteps);
      // printf("nrec = %d\n", src_rec.vec_nrec.at(iShot));
      // exit(0);
      // the default type of bin file is float32.
    }
    
#ifdef VERBOSE
    auto finishSrc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSrc = finishSrc - startSrc;
    std::cout << "Obs data saving time: " << elapsedSrc.count() << " second(s)"
              << std::endl;
#endif
  }

  // output residual
  if (para.if_res()) {
#ifdef VERBOSE
    std::cout << "Total l2 residual = " << std::to_string(h_l2Obj) << std::endl;
    std::cout << "calc_id = " << calc_id << std::endl;
#endif
    *misfit = h_l2Obj;
  }

  free(h_l2Obj_temp_x);
  free(h_l2Obj_temp_y);
  free(h_l2Obj_temp_z);

  free(fLambda);
  free(fMu);
  free(fRho);

  // destroy the streams
  for (int iShot = 0; iShot < group_size; iShot++)
    CHECK(cudaStreamDestroy(streams[iShot]));

  cudaFree(d_vz);
  cudaFree(d_vx);
  cudaFree(d_vy);
  cudaFree(d_szz);
  cudaFree(d_sxx);
  cudaFree(d_sxz);
  cudaFree(d_syy);
  cudaFree(d_syz);
  cudaFree(d_sxy);
  cudaFree(d_vz_adj);
  cudaFree(d_vx_adj);
  cudaFree(d_vy_adj);
  cudaFree(d_szz_adj);
  cudaFree(d_sxx_adj);
  cudaFree(d_sxz_adj);
  cudaFree(d_syy_adj);
  cudaFree(d_syz_adj);
  cudaFree(d_sxy_adj);
  cudaFree(d_mem_dvz_dz);
  cudaFree(d_mem_dvz_dx);
  cudaFree(d_mem_dvz_dy);
  cudaFree(d_mem_dvx_dz);
  cudaFree(d_mem_dvx_dx);
  cudaFree(d_mem_dvx_dy);
  cudaFree(d_mem_dvy_dz);
  cudaFree(d_mem_dvy_dx);
  cudaFree(d_mem_dvy_dy);
  cudaFree(d_mem_dszz_dz);
  cudaFree(d_mem_dsxx_dx);
  cudaFree(d_mem_dsyy_dy);
  cudaFree(d_mem_dsxz_dz);
  cudaFree(d_mem_dsxz_dx);
  cudaFree(d_mem_dsyz_dy);
  cudaFree(d_mem_dsyz_dz);
  cudaFree(d_mem_dsxy_dx);
  cudaFree(d_mem_dsxy_dy);
  cudaFree(d_l2Obj_temp_x);
  cudaFree(d_l2Obj_temp_y);
  cudaFree(d_l2Obj_temp_z);
  cudaFree(d_gauss_amp);
#ifdef VERBOSE
  std::cout << "Done!" << std::endl;
#endif
}
