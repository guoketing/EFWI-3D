#include "utilities.h"

__global__ void stress(float *d_vz, float *d_vx, float *d_vy, float *d_szz, \
	float *d_sxx, float *d_sxz, float *d_syy, float *d_sxy, float *d_syz, \
	float *d_mem_dvz_dz, float *d_mem_dvz_dx, float *d_mem_dvz_dy, \
	float *d_mem_dvx_dz, float *d_mem_dvx_dx, float *d_mem_dvx_dy, \
	float *d_mem_dvy_dz, float *d_mem_dvy_dx, float *d_mem_dvy_dy, \
	float *d_Lambda, float *d_Mu, float *d_Den, \
	float *d_K_z, float *d_a_z, float *d_b_z, float *d_K_z_half, float *d_a_z_half, float *d_b_z_half, \
	float *d_K_x, float *d_a_x, float *d_b_x, float *d_K_x_half, float *d_a_x_half, float *d_b_x_half, \
	float *d_K_y, float *d_a_y, float *d_b_y, float *d_K_y_half, float *d_a_y_half, float *d_b_y_half, \
	int nz, int nx, int ny, float dt, float dz, float dx, float dy, int nPml, bool isFor, \
	float *d_szz_adj, float *d_sxx_adj, float *d_sxz_adj, float *d_syy_adj, float *d_sxy_adj, float *d_syz_adj, \
	float *d_LambdaGrad, float *d_MuGrad){


  int gidx = blockIdx.x*blockDim.x + threadIdx.x;
  int gidy = blockIdx.y*blockDim.y + threadIdx.y;
  int gidz = blockIdx.z*blockDim.z + threadIdx.z;

  int id = (gidy)*(nx*nz)+(gidx)*nz+(gidz);

  float dvz_dz = 0.0;
  float dvx_dx = 0.0;
  float dvy_dy = 0.0;
  float dvx_dz = 0.0;
  float dvx_dy = 0.0;
  float dvz_dx = 0.0;
  float dvz_dy = 0.0;
  float dvy_dx = 0.0;
  float dvy_dz = 0.0;

  float c1 = 9.0/8.0;
  float c2 = 1.0/24.0;

  if (isFor) {

		if(gidz>=2 && gidz<=nz-3 && gidx>=2 && gidx<=nx-3 && gidy>=2 && gidy<=ny-3) {

		  dvz_dz = (c1*(d_vz[id]-d_vz[id-1]) - c2*(d_vz[id+1]-d_vz[id-2]))/dz;
		  dvx_dx = (c1*(d_vx[id]-d_vx[id-nz]) - c2*(d_vx[id+nz]-d_vx[id-2*nz]))/dx;
		  dvy_dy = (c1*(d_vy[id]-d_vy[id-nz*nx]) - c2*(d_vy[id+nz*nx]-d_vy[id-2*nz*nx]))/dy;

		    if(gidz<nPml || (gidz>nz-nPml-1)){
			  d_mem_dvz_dz[id] = d_b_z[gidz]*d_mem_dvz_dz[id] + d_a_z[gidz]*dvz_dz;
			  dvz_dz = dvz_dz / d_K_z[gidz] + d_mem_dvz_dz[id];
			}
			if(gidx<nPml || gidx>nx-nPml-1){
			  d_mem_dvx_dx[id] = d_b_x[gidx]*d_mem_dvx_dx[id] + d_a_x[gidx]*dvx_dx;
			  dvx_dx = dvx_dx / d_K_x[gidx] + d_mem_dvx_dx[id];
			}
			if(gidy<nPml || gidy>ny-nPml-1){
			  d_mem_dvy_dy[id] = d_b_y[gidy]*d_mem_dvy_dy[id] + d_a_y[gidy]*dvy_dy;
			  dvy_dy = dvy_dy / d_K_y[gidy] + d_mem_dvy_dy[id];
			}

		  d_szz[id] += ((d_Lambda[id]+2.0*d_Mu[id])*dvz_dz + d_Lambda[id]*dvx_dx + d_Lambda[id]*dvy_dy) * dt;
		  d_sxx[id] += (d_Lambda[id]*dvz_dz + (d_Lambda[id]+2.0*d_Mu[id])*dvx_dx + d_Lambda[id]*dvy_dy) * dt;
		  d_syy[id] += (d_Lambda[id]*dvz_dz + d_Lambda[id]*dvx_dx + (d_Lambda[id]+2.0*d_Mu[id])*dvy_dy) * dt;

		  dvx_dz = (c1*(d_vx[id+1]-d_vx[id]) - c2*(d_vx[id+2]-d_vx[id-1]))/dz;
		  dvx_dy = (c1*(d_vx[id+nz*nx]-d_vx[id]) - c2*(d_vx[id+2*nz*nx]-d_vx[id-nz*nx]))/dy;

		  dvz_dx = (c1*(d_vz[id+nz]-d_vz[id]) - c2*(d_vz[id+2*nz]-d_vz[id-nz]))/dx;
		  dvz_dy = (c1*(d_vz[id+nz*nx]-d_vz[id]) - c2*(d_vz[id+2*nz*nx]-d_vz[id-nz*nx]))/dy;

		  dvy_dx = (c1*(d_vy[id+nz]-d_vy[id]) - c2*(d_vy[id+2*nz]-d_vy[id-nz]))/dx;
		  dvy_dz = (c1*(d_vy[id+1]-d_vy[id]) - c2*(d_vy[id+2]-d_vy[id-1]))/dz;

		  if(gidz<nPml || (gidz>nz-nPml-1)){
			  d_mem_dvx_dz[id] = d_b_z_half[gidz]*d_mem_dvx_dz[id] + d_a_z_half[gidz]*dvx_dz;
			  dvx_dz = dvx_dz / d_K_z_half[gidz] + d_mem_dvx_dz[id];

			  d_mem_dvy_dz[id] = d_b_z_half[gidz]*d_mem_dvy_dz[id] + d_a_z_half[gidz]*dvy_dz;
			  dvy_dz = dvy_dz / d_K_z_half[gidz] + d_mem_dvy_dz[id];
			}

		  if(gidx<nPml || gidx>nx-nPml-1){
			  d_mem_dvz_dx[id] = d_b_x_half[gidx]*d_mem_dvz_dx[id] + d_a_x_half[gidx]*dvz_dx;
			  dvz_dx = dvz_dx / d_K_x_half[gidx] + d_mem_dvz_dx[id];

			  d_mem_dvy_dx[id] = d_b_x_half[gidx]*d_mem_dvy_dx[id] + d_a_x_half[gidx]*dvy_dx;
			  dvy_dx = dvy_dx / d_K_x_half[gidx] + d_mem_dvy_dx[id];
			}

			if(gidy<nPml || gidy>ny-nPml-1){
			  d_mem_dvz_dy[id] = d_b_y_half[gidy]*d_mem_dvz_dy[id] + d_a_y_half[gidy]*dvz_dy;
			  dvz_dy = dvz_dy / d_K_y_half[gidy] + d_mem_dvz_dy[id];

			  d_mem_dvx_dy[id] = d_b_y_half[gidy]*d_mem_dvx_dy[id] + d_a_y_half[gidy]*dvx_dy;
			  dvx_dy = dvx_dy / d_K_y_half[gidy] + d_mem_dvx_dy[id];
			}

		  d_sxz[id] = d_sxz[id] + d_Mu[id] * (dvx_dz + dvz_dx) * dt;
		  d_sxy[id] = d_sxy[id] + d_Mu[id] * (dvx_dy + dvy_dx) * dt;
		  d_syz[id] = d_syz[id] + d_Mu[id] * (dvy_dz + dvz_dy) * dt;
		}
		else{
			return;
		}
	}

	else {

		// ========================================BACKWARD PROPAGATION====================================
		if(gidz>=nPml && gidz<=nz-1-nPml && gidx>=nPml && gidx<=nx-1-nPml && gidy>=nPml && gidy<=ny-1-nPml) {

		  dvz_dz = (c1*(d_vz[id]-d_vz[id-1]) - c2*(d_vz[id+1]-d_vz[id-2]))/dz;
		  dvx_dx = (c1*(d_vx[id]-d_vx[id-nz]) - c2*(d_vx[id+nz]-d_vx[id-2*nz]))/dx;
          dvy_dy = (c1*(d_vy[id]-d_vy[id-nz*nx]) - c2*(d_vy[id+nz*nx]-d_vy[id-2*nz*nx]))/dy;

		  d_szz[id] -= ((d_Lambda[id]+2.0*d_Mu[id])*dvz_dz + d_Lambda[id]*dvx_dx + d_Lambda[id]*dvy_dy) * dt;
		  d_sxx[id] -= (d_Lambda[id]*dvz_dz + (d_Lambda[id]+2.0*d_Mu[id])*dvx_dx + d_Lambda[id]*dvy_dy) * dt;
          d_syy[id] -= (d_Lambda[id]*dvz_dz + d_Lambda[id]*dvx_dx + (d_Lambda[id]+2.0*d_Mu[id])*dvy_dy) * dt;

		  dvx_dz = (c1*(d_vx[id+1]-d_vx[id]) - c2*(d_vx[id+2]-d_vx[id-1]))/dz;
		  dvx_dy = (c1*(d_vx[id+nz*nx]-d_vx[id]) - c2*(d_vx[id+2*nz*nx]-d_vx[id-nz*nx]))/dy;

		  dvz_dx = (c1*(d_vz[id+nz]-d_vz[id]) - c2*(d_vz[id+2*nz]-d_vz[id-nz]))/dx;
		  dvz_dy = (c1*(d_vz[id+nz*nx]-d_vz[id]) - c2*(d_vz[id+2*nz*nx]-d_vz[id-nz*nx]))/dy;

		  dvy_dx = (c1*(d_vy[id+nz]-d_vy[id]) - c2*(d_vy[id+2*nz]-d_vy[id-nz]))/dx;
		  dvy_dz = (c1*(d_vy[id+1]-d_vy[id]) - c2*(d_vy[id+2]-d_vy[id-1]))/dz;

		  d_sxz[id] -= d_Mu[id] * (dvx_dz + dvz_dx) * dt;
		  d_sxy[id] -= d_Mu[id] * (dvx_dy + dvy_dx) * dt;
		  d_syz[id] -= d_Mu[id] * (dvy_dz + dvz_dy) * dt;
			
			// computate the kernels of lame parameters
			d_LambdaGrad[id] += -(d_szz_adj[id]+d_sxx_adj[id]+d_syy_adj[id]) * (dvz_dz+dvx_dx+dvy_dy) * dt;

			d_MuGrad[id] += -(2.0*d_szz_adj[id]*dvz_dz+2.0*d_sxx_adj[id]*dvx_dx+\
			                            2.0*d_syy_adj[id]*dvy_dy+d_sxz_adj[id]*(dvx_dz+dvz_dx)+\
			                            d_syz_adj[id]*(dvy_dz+dvz_dy)+d_sxy_adj[id]*(dvy_dx+dvx_dy))*dt;
		}
		else{
			return;
		}
	}
}
