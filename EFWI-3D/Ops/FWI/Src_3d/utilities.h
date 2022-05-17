#ifndef UTILITIES_H__
#define UTILITIES_H__

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "cufft.h"
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"

#define PI (3.141592653589793238462643383279502884197169)


#define DIVCONST 1e-9

__constant__ float coef[2];

#define CHECK(call)                                                   \
  {                                                                      \
    const cudaError_t error = call;                                      \
    if (error != cudaSuccess) {                                          \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                           \
    }                                                                    \
  }

void fileBinLoad(float *h_bin, int size, std::string fname);

void fileBinWrite(float *h_bin, int size, std::string fname);

void fileBinWriteDouble(double *h_bin, int size, std::string fname);

void initialArray(float *ip, int size, float value);

void initialArray(double *ip, int size, double value);

__global__ void initialArrayGPU(float *ip, int nz, int nx, int ny, float value);

__global__ void initial2DArrayGPU(float *ip, int nx, int ny, float value);

__global__ void assignArrayGPU(float *ip_in, float *ip_out, int nx, int ny, int nz);

void displayArray(std::string s, float *ip, int nx, int ny);


__global__ void velInit(float *d_Lambda, float *d_Mu, float *d_Rho, float *d_Vp,
                        float *d_Vs, int nz, int nx, int ny);

__global__ void aveMuInit(float *d_Mu, float *d_ave_Mu, int nz, int nx, int ny);

__global__ void aveBycInit(float *d_Rho, float *d_Byc, int nz, int nx, int ny);

__global__ void gpuMinus(float *d_out, float *d_in1, float *d_in2, int nx,
                         int ny);

__global__ void cuda_cal_objective(float *obj, float *err, int ng);

float cal_objective(float *array, int N);

float compVpAve(float *array, int N);

void compCourantNumber(float *h_Vp, int size, float dt, float dz, float dx, float dy);

void cpmlInit(float *K, float *a, float *b, float *K_half, float *a_half,
              float *b_half, int N, int nPml, float dh, float f0, float dt,
              float VpAve);

__global__ void stress(
    float *d_vz, float *d_vx, float *d_vy, float *d_szz,
    float *d_sxx, float *d_sxz, float *d_syy, float *d_sxy,float *d_syz,
    float *d_mem_dvz_dz, float *d_mem_dvz_dx, float *d_mem_dvz_dy,
    float *d_mem_dvx_dz, float *d_mem_dvx_dx, float *d_mem_dvx_dy,
    float *d_mem_dvy_dz, float *d_mem_dvy_dx, float *d_mem_dvy_dy,
    float *d_Lambda, float *d_Mu, float *d_Rho,
    float *d_K_z, float *d_a_z, float *d_b_z, float *d_K_z_half, float *d_a_z_half, float *d_b_z_half,
    float *d_K_x, float *d_a_x, float *d_b_x, float *d_K_x_half, float *d_a_x_half, float *d_b_x_half,
    float *d_K_y, float *d_a_y, float *d_b_y, float *d_K_y_half, float *d_a_y_half, float *d_b_y_half,
    int nz, int nx, int ny, float dt, float dz, float dx, float dy, int nPml, bool isFor,
    float *d_szz_adj, float *d_sxx_adj, float *d_sxz_adj, float *d_syy_adj,
    float *d_sxy_adj, float *d_syz_adj, float *d_LambdaGrad, float *d_MuGrad);

__global__ void velocity(
    float *d_vz, float *d_vx, float *d_vy, float *d_szz, float *d_sxx, float *d_sxz,
    float *d_syy,float *d_sxy,float *d_syz, float *d_mem_dszz_dz, float *d_mem_dsxz_dx, float *d_mem_dsxz_dz,
    float *d_mem_dsxx_dx, float *d_mem_dsxy_dx, float *d_mem_dsxy_dy, float *d_mem_dsyy_dy,
    float *d_mem_dsyz_dy, float *d_mem_dsyz_dz, float *d_Lambda, float *d_Mu, float *d_Byc,
    float *d_K_z, float *d_a_z, float *d_b_z,
    float *d_K_z_half, float *d_a_z_half, float *d_b_z_half, float *d_K_x,
    float *d_a_x, float *d_b_x, float *d_K_x_half, float *d_a_x_half,
    float *d_b_x_half, float *d_K_y, float *d_a_y, float *d_b_y,
	float *d_K_y_half, 	float *d_a_y_half, float *d_b_y_half,
	int nz, int nx, int ny, float dt, float dz, float dx, float dy, int nPml,
  bool isFor, float *d_vz_adj, float *d_vx_adj, float *d_vy_adj, float *d_RhoGrad);


__global__ void velocity_adj(
  float *d_vz, float *d_vx, float *d_vy, float *d_szz, float *d_sxx, float *d_sxz, 
  float *d_syy, float *d_sxy, float *d_syz, float *d_mem_dszz_dz, float *d_mem_dsxz_dx, 
  float *d_mem_dsxz_dz, float *d_mem_dsxx_dx, float *d_mem_dsxy_dx, float *d_mem_dsxy_dy,
  float *d_mem_dsyy_dy, float *d_mem_dsyz_dy, float *d_mem_dsyz_dz,
  float *d_mem_dvz_dz, float *d_mem_dvz_dx, float *d_mem_dvz_dy, float *d_mem_dvx_dz, float *d_mem_dvx_dx,
  float *d_mem_dvx_dy, float *d_mem_dvy_dz, float *d_mem_dvy_dx,float *d_mem_dvy_dy,
  float *d_Lambda, float *d_Mu, float *d_Rho, float *d_Byc,
  float *d_K_z_half, float *d_a_z_half, float *d_b_z_half,
	float *d_K_x_half, float *d_a_x_half, float *d_b_x_half,
	float *d_K_y_half, float *d_a_y_half, float *d_b_y_half,
	float *d_K_z, float *d_a_z, float *d_b_z,
	float *d_K_x, float *d_a_x, float *d_b_x, 
	float *d_K_y, float *d_a_y, float *d_b_y, 
	int nz, int nx, int ny, float dt, float dz, float dx, float dy, int nPml);


__global__ void stress_adj(
  float *d_vz, float *d_vx, float *d_vy, float *d_szz, float *d_sxx, float *d_sxz, 
  float *d_syy, float *d_sxy, float *d_syz, float *d_mem_dszz_dz, float *d_mem_dsxz_dx, 
  float *d_mem_dsxz_dz, float *d_mem_dsxx_dx, float *d_mem_dsxy_dx, float *d_mem_dsxy_dy,
  float *d_mem_dsyy_dy, float *d_mem_dsyz_dy, float *d_mem_dsyz_dz, 
  float *d_mem_dvz_dz, float *d_mem_dvz_dx, float *d_mem_dvz_dy, float *d_mem_dvx_dz, float *d_mem_dvx_dx, 
  float *d_mem_dvx_dy, float *d_mem_dvy_dz, float *d_mem_dvy_dx,float *d_mem_dvy_dy, 
  float *d_Lambda, float *d_Mu, float *d_Rho, float *d_Byc,
  float *d_K_z_half, float *d_a_z_half, float *d_b_z_half,
	float *d_K_x_half, float *d_a_x_half, float *d_b_x_half, 
	float *d_K_y_half, float *d_a_y_half, float *d_b_y_half, 
	float *d_K_z, float *d_a_z, float *d_b_z, 
	float *d_K_x, float *d_a_x, float *d_b_x, 
	float *d_K_y, float *d_a_y, float *d_b_y, 
	int nz, int nx, int ny, float dt, float dz, float dx, float dy, int nPml);

__global__ void velocity_adj(
    float *d_vz, float *d_vx, float *d_szz, float *d_sxx, float *d_sxz,
    float *d_mem_dszz_dz, float *d_mem_dsxz_dx, float *d_mem_dsxz_dz,
    float *d_mem_dsxx_dx, float *d_mem_dvz_dz, float *d_mem_dvz_dx,
    float *d_mem_dvx_dz, float *d_mem_dvx_dx, float *d_Lambda, float *d_Mu,
    float *d_ave_Mu, float *d_Rho, float *d_ave_Byc_a, float *d_ave_Byc_b,
    float *d_K_z_half, float *d_a_z_half, float *d_b_z_half, float *d_K_x_half,
    float *d_a_x_half, float *d_b_x_half, float *d_K_z, float *d_a_z,
    float *d_b_z, float *d_K_x, float *d_a_x, float *d_b_x, int nz, int nx,
    float dt, float dz, float dx, int nPml);


__global__ void src_rec_gauss_amp(float *gauss_amp, int nz, int nx, int ny);

__global__ void add_source(float *d_szz, float *d_sxx, float *d_syy, float amp, int nz, int nx, int ny,
                           bool isFor, int z_loc, int x_loc, int y_loc, float dt, float *gauss_amp);

__global__ void recording(float *d_vz, float *d_vx, float *d_vy, int nx, int ny, float *d_data_x, float *d_data_y, float *d_data_z,
                          int iShot, int it, int nSteps, int nrec, int *d_z_rec, int *d_x_rec, int *d_y_rec);

__global__ void res_injection(float *d_vx_adj, float *d_vy_adj, float *d_vz_adj, int nz, int nx, 
                              float *d_res_x, float *d_res_y, float *d_res_z, int it, float dt, int nSteps,
                              int nrec, int *d_z_rec, int *d_x_rec, int *d_y_rec);

__global__ void source_grad(float *d_szz_adj, float *d_sxx_adj, float *d_syy_adj, int nz, int nx,
                            float *d_StfGrad, int it, float dt, int z_src,
                            int x_src,  int y_src, double rxz, double rxy);


__global__ void from_bnd(float *d_field, float *d_bnd, int nz, int nx, int ny,
                         int nzBnd, int nxBnd, int nyBnd, int len_Bnd_vec, int nLayerStore,
                         int indT, int nPml, int nSteps);

__global__ void to_bnd(float *d_field, float *d_bnd, int nz, int nx, int ny, int nzBnd,
                       int nxBnd, int nyBnd, int len_Bnd_vec, int nLayerStore, int indT,
                       int nPml, int nSteps);




__global__ void cuda_window(int nt, int nrec, float dt, float *d_win_start,
                            float *d_win_end, float *d_weights,
                            float src_weight, float ratio, float *data);

__global__ void cuda_window(int nt, int nrec, float dt, float ratio,
                            float *data);



#endif