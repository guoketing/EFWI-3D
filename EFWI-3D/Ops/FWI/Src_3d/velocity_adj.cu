#include<stdio.h>

__global__ void velocity_adj(
  float *d_vz, float *d_vx, float *d_vy, float *d_szz, float *d_sxx, float *d_sxz, \
  float *d_syy, float *d_sxy, float *d_syz, float *d_mem_dszz_dz, float *d_mem_dsxz_dx, \
  float *d_mem_dsxz_dz, float *d_mem_dsxx_dx, float *d_mem_dsxy_dx, float *d_mem_dsxy_dy,\
  float *d_mem_dsyy_dy, float *d_mem_dsyz_dy, float *d_mem_dsyz_dz, \
  float *d_mem_dvz_dz, float *d_mem_dvz_dx, float *d_mem_dvz_dy, float *d_mem_dvx_dz, float *d_mem_dvx_dx, \
  float *d_mem_dvx_dy, float *d_mem_dvy_dz, float *d_mem_dvy_dx,float *d_mem_dvy_dy, \
  float *d_Lambda, float *d_Mu, float *d_Den, float *d_Byc, \
  float *d_K_z_half, float *d_a_z_half, float *d_b_z_half, \
	float *d_K_x_half, float *d_a_x_half, float *d_b_x_half, \
	float *d_K_y_half, float *d_a_y_half, float *d_b_y_half, \
	float *d_K_z, float *d_a_z, float *d_b_z, \
	float *d_K_x, float *d_a_x, float *d_b_x, \
	float *d_K_y, float *d_a_y, float *d_b_y, \
	int nz, int nx, int ny, float dt, float dz, float dx, float dy, int nPml){
	
	int gidx = blockIdx.x*blockDim.x + threadIdx.x;
    int gidy = blockIdx.y*blockDim.y + threadIdx.y;
    int gidz = blockIdx.z*blockDim.z + threadIdx.z;

	int id = gidy * nx*nz + gidx * nz + gidz;

    float dpsixx_dx = 0.0;
	float dsxx_dx = 0.0;
	float dszz_dx = 0.0;
	float dsyy_dx = 0.0;
	float dpsixy_dy = 0.0;
	float dsxy_dy = 0.0;
	float dpsixz_dz = 0.0;
	float dsxz_dz = 0.0;

	float dpsiyy_dy = 0.0;
    float dsyy_dy = 0.0;
	float dsxx_dy = 0.0;
	float dszz_dy = 0.0;
	float dpsiyx_dx = 0.0;
    float dsxy_dx = 0.0;
    float dpsiyz_dz = 0.0;
    float dsyz_dz = 0.0;

    float dpsizz_dz = 0.0;
	float dszz_dz = 0.0;
	float dsxx_dz = 0.0;
	float dsyy_dz = 0.0;
	float dpsizy_dy = 0.0;
	float dsyz_dy = 0.0;
	float dpsizx_dx = 0.0;
	float dsxz_dx = 0.0;

    float c1 = 9.0/8.0;
	float c2 = 1.0/24.0;
	
	float lambda = d_Lambda[id];
    float mu = d_Mu[id];
	float Byc = d_Byc[id];

    if(gidz>=2 && gidz<=nz-3 && gidx>=2 && gidx<=nx-3 && gidy>=2 && gidy<=ny-3) {

		// update vx
		dpsixx_dx = (-c1*(d_mem_dvx_dx[id+nz]-d_mem_dvx_dx[id]) + c2*(d_mem_dvx_dx[id+2*nz]-d_mem_dvx_dx[id-nz]))/dx;
		dszz_dx = (-c1*(d_szz[id+nz]-d_szz[id]) + c2*(d_szz[id+2*nz]-d_szz[id-nz]))/dx;
		dsxx_dx = (-c1*(d_sxx[id+nz]-d_sxx[id]) + c2*(d_sxx[id+2*nz]-d_sxx[id-nz]))/dx;
		dsyy_dx = (-c1*(d_syy[id+nz]-d_syy[id]) + c2*(d_syy[id+2*nz]-d_syy[id-nz]))/dx;

		dpsixz_dz = (-c1*(d_mem_dvx_dz[id]-d_mem_dvx_dz[id-1]) + c2*(d_mem_dvx_dz[id+1]-d_mem_dvx_dz[id-2]))/dz;
		dsxz_dz = (-c1*(d_sxz[id]-d_sxz[id-1]) + c2*(d_sxz[id+1]-d_sxz[id-2]))/dz;
        dpsixy_dy = (-c1*(d_mem_dvx_dy[id]-d_mem_dvx_dy[id-nz*nx]) + c2*(d_mem_dvx_dy[id+nz*nx]-d_mem_dvx_dy[id-2*nz*nx]))/dy;
        dsxy_dy = (-c1*(d_sxy[id]-d_sxy[id-nz*nx]) + c2*(d_sxy[id+nz*nx]-d_sxy[id-2*nz*nx]))/dy;

		d_vx[id] += (d_a_x[gidx]*dpsixx_dx + (lambda+2.0*mu)*dsxx_dx/d_K_x[gidx]*dt 
		         + lambda*dszz_dx/d_K_x[gidx]*dt + lambda*dsyy_dx/d_K_x[gidx]*dt
		         + mu/d_K_z_half[gidz]*dsxz_dz*dt + d_a_z_half[gidz]*dpsixz_dz
				 + mu/d_K_y_half[gidy]*dsxy_dy*dt + d_a_y_half[gidy]*dpsixy_dy);

		//update phi_xx_x , phi_xz_z phi_xy_y
		if(gidx<nPml || (gidx>nx-nPml-1)){
			d_mem_dsxx_dx[id] = d_b_x_half[gidx]*d_mem_dsxx_dx[id] + Byc*d_vx[id]*dt;
		}
		if(gidz<nPml || (gidz>nz-nPml-1)){
			d_mem_dsxz_dz[id] = d_b_z[gidz]*d_mem_dsxz_dz[id] + Byc*d_vx[id]*dt;
		}
		if(gidy<nPml || (gidy>ny-nPml-1)){
			d_mem_dsxy_dy[id] = d_b_y[gidy]*d_mem_dsxy_dy[id] + Byc*d_vx[id]*dt;
		}

	  	// update vz
		dpsizz_dz = (-c1*(d_mem_dvz_dz[id+1]-d_mem_dvz_dz[id]) + c2*(d_mem_dvz_dz[id+2]-d_mem_dvz_dz[id-1]))/dz;
		dszz_dz = (-c1*(d_szz[id+1]-d_szz[id]) + c2*(d_szz[id+2]-d_szz[id-1]))/dz;
		dsxx_dz = (-c1*(d_sxx[id+1]-d_sxx[id]) + c2*(d_sxx[id+2]-d_sxx[id-1]))/dz;
		dsyy_dz = (-c1*(d_syy[id+1]-d_syy[id]) + c2*(d_syy[id+2]-d_syy[id-1]))/dz;
		
		dpsizx_dx = (-c1*(d_mem_dvz_dx[id]-d_mem_dvz_dx[id-nz]) + c2*(d_mem_dvz_dx[id+nz]-d_mem_dvz_dx[id-2*nz]))/dx;
		dsxz_dx = (-c1*(d_sxz[id]-d_sxz[id-nz]) + c2*(d_sxz[id+nz]-d_sxz[id-2*nz]))/dx;

		dpsizy_dy = (-c1*(d_mem_dvz_dy[id]-d_mem_dvz_dy[id-nz*nx]) + c2*(d_mem_dvz_dy[id+nz*nx]-d_mem_dvz_dy[id-2*nz*nx]))/dy;
		dsyz_dy = (-c1*(d_syz[id]-d_syz[id-nz*nx]) + c2*(d_syz[id+nz*nx]-d_syz[id-2*nz*nx]))/dy;

		d_vz[id] += (d_a_z[gidz]*dpsizz_dz + (lambda+2.0*mu)*dszz_dz/d_K_z[gidz]*dt
			+ lambda*dsxx_dz/d_K_z[gidz]*dt + lambda*dsyy_dz/d_K_z[gidz]*dt
			+ mu/d_K_y_half[gidy]*dsyz_dy*dt + d_a_y_half[gidy]*dpsizy_dy
			+ mu/d_K_x_half[gidx]*dsxz_dx*dt + d_a_x_half[gidx]*dpsizx_dx);

		// update phi_xz_x, phi_zz_z && phi_yz_y
		if(gidx<nPml || (gidx>nx-nPml-1)){
			d_mem_dsxz_dx[id] = d_b_x[gidx]*d_mem_dsxz_dx[id] + Byc*d_vz[id]*dt;
		}
		if(gidz<nPml || (gidz>nz-nPml-1)){
			d_mem_dszz_dz[id] = d_b_z_half[gidz]*d_mem_dszz_dz[id] + Byc*d_vz[id]*dt;
		}
		if(gidy<nPml || (gidy>ny-nPml-1)){
			d_mem_dsyz_dy[id] = d_b_y[gidy]*d_mem_dsyz_dy[id] + Byc*d_vz[id]*dt;
		}

		// update vy

		dpsiyy_dy = (-c1*(d_mem_dvy_dy[id+nz*nx]-d_mem_dvy_dy[id]) + c2*(d_mem_dvy_dy[id+2*nz*nx]-d_mem_dvy_dy[id-nz*nx]))/dy;
		dsyy_dy = (-c1*(d_syy[id+nz*nx]-d_syy[id]) + c2*(d_syy[id+2*nz*nx]-d_syy[id-nz*nx]))/dy;
		dsxx_dy = (-c1*(d_sxx[id+nz*nx]-d_sxx[id]) + c2*(d_sxx[id+2*nz*nx]-d_sxx[id-nz*nx]))/dy;
		dszz_dy = (-c1*(d_szz[id+nz*nx]-d_szz[id]) + c2*(d_szz[id+2*nz*nx]-d_szz[id-nz*nx]))/dy;
		
		dpsiyx_dx = (-c1*(d_mem_dvy_dx[id]-d_mem_dvy_dx[id-nz]) + c2*(d_mem_dvy_dx[id+nz]-d_mem_dvy_dx[id-2*nz]))/dx;
		dsxy_dx = (-c1*(d_sxy[id]-d_sxy[id-nz]) + c2*(d_sxy[id+nz]-d_sxy[id-2*nz]))/dx;

		dpsiyz_dz = (-c1*(d_mem_dvy_dz[id]-d_mem_dvy_dz[id-1]) + c2*(d_mem_dvy_dz[id+1]-d_mem_dvy_dz[id-2]))/dz;
		dsyz_dz = (-c1*(d_syz[id]-d_syz[id-1]) + c2*(d_syz[id+1]-d_syz[id-2]))/dz;

		d_vy[id] += (d_a_y[gidy]*dpsiyy_dy + (lambda+2.0*mu)*dsyy_dy/d_K_y[gidy]*dt
			+ lambda*dsxx_dy/d_K_y[gidy]*dt + lambda*dszz_dy/d_K_y[gidy]*dt
			+ mu/d_K_x_half[gidx]*dsxy_dx*dt + d_a_x_half[gidx]*dpsiyx_dx
			+ mu/d_K_z_half[gidz]*dsyz_dz*dt + d_a_z_half[gidz]*dpsiyz_dz);

		// update phi_xy_x, phi_yz_z && phi_yy_y
		if(gidx<nPml || (gidx>nx-nPml-1)){
			d_mem_dsxy_dx[id] = d_b_x[gidx]*d_mem_dsxy_dx[id] + Byc*d_vy[id]*dt;
		}
		if(gidz<nPml || (gidz>nz-nPml-1)){
			d_mem_dsyz_dz[id] = d_b_z_half[gidz]*d_mem_dsyz_dz[id] + Byc*d_vy[id]*dt;
		}
		if(gidy<nPml || (gidy>ny-nPml-1)){
			d_mem_dsyy_dy[id] = d_b_y[gidy]*d_mem_dsyy_dy[id] + Byc*d_vy[id]*dt;
		}
	}

	else {
		return;
	}

}