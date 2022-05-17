#include<stdio.h>

__global__ void velocity(float *d_vz, float *d_vx, float *d_vy, float *d_szz, \
	float *d_sxx, float *d_sxz, float *d_syy, float *d_sxy, float *d_syz, float *d_mem_dszz_dz, float *d_mem_dsxz_dx, \
	float *d_mem_dsxz_dz, float *d_mem_dsxx_dx, float *d_mem_dsxy_dx, float *d_mem_dsxy_dy,\
	float *d_mem_dsyy_dy, float *d_mem_dsyz_dy, float *d_mem_dsyz_dz, float *d_Lambda, float *d_Mu, \
	float *d_Byc, float *d_K_z, float *d_a_z, float *d_b_z, \
	float *d_K_z_half, 	float *d_a_z_half, float *d_b_z_half, float *d_K_x, float *d_a_x, \
	float *d_b_x, float *d_K_x_half, float *d_a_x_half, float *d_b_x_half, \
	float *d_K_y, float *d_a_y, float *d_b_y, \
	float *d_K_y_half, 	float *d_a_y_half, float *d_b_y_half,\
	int nz, int nx, int ny, float dt, float dz, float dx, float dy, int nPml, bool isFor, \
	float *d_vz_adj, float *d_vx_adj, float *d_vy_adj, float *d_DenGrad){

	int gidx = blockIdx.x*blockDim.x + threadIdx.x;
    int gidy = blockIdx.y*blockDim.y + threadIdx.y;
    int gidz = blockIdx.z*blockDim.z + threadIdx.z;

	int id = (gidy)*(nx*nz)+(gidx)*nz+(gidz);


  float dszz_dz = 0.0;
  float dsxz_dx = 0.0;
  float dsxz_dz = 0.0;
  float dsxx_dx = 0.0;
  float dsxy_dx = 0.0;
  float dsxy_dy = 0.0;
  float dsyy_dy = 0.0;
  float dsyz_dy = 0.0;
  float dsyz_dz = 0.0;

  float c1 = 9.0/8.0;
  float c2 = 1.0/24.0;

  if (isFor) {
	  if(gidz>=2 && gidz<=nz-3 && gidx>=2 && gidx<=nx-3 && gidy>=2 && gidy<=ny-3) {
		    // update vz
			dszz_dz = (c1*(d_szz[id+1]-d_szz[id]) - c2*(d_szz[id+2]-d_szz[id-1]))/dz;
			dsxz_dx = (c1*(d_sxz[id]-d_sxz[id-nz]) - c2*(d_sxz[id+nz]-d_sxz[id-2*nz]))/dx;
			dsyz_dy = (c1*(d_syz[id]-d_syz[id-nz*nx]) - c2*(d_syz[id+nz*nx]-d_syz[id-2*nz*nx]))/dy;

			if(gidz<nPml || (gidz>nz-nPml-1)){
				d_mem_dszz_dz[id] = d_b_z_half[gidz]*d_mem_dszz_dz[id] + d_a_z_half[gidz]*dszz_dz;
				dszz_dz = dszz_dz / d_K_z_half[gidz] + d_mem_dszz_dz[id];
		    }
		    if(gidx<nPml || gidx>nx-nPml){
				d_mem_dsxz_dx[id] = d_b_x[gidx]*d_mem_dsxz_dx[id] + d_a_x[gidx]*dsxz_dx;
				dsxz_dx = dsxz_dx / d_K_x[gidx] + d_mem_dsxz_dx[id];
			}
			if(gidy<nPml || gidy>ny-nPml){
				d_mem_dsyz_dy[id] = d_b_y[gidy]*d_mem_dsyz_dy[id] + d_a_y[gidy]*dsyz_dy;
				dsyz_dy = dsyz_dy / d_K_y[gidy] + d_mem_dsyz_dy[id];
			}

			d_vz[id] += (dszz_dz + dsxz_dx + dsyz_dy) * d_Byc[id] * dt;

			// update vx
			dsxz_dz = (c1*(d_sxz[id]-d_sxz[id-1]) - c2*(d_sxz[id+1]-d_sxz[id-2]))/dz;
			dsxx_dx = (c1*(d_sxx[id+nz]-d_sxx[id]) - c2*(d_sxx[id+2*nz]-d_sxx[id-nz]))/dx;
			dsxy_dy = (c1*(d_sxy[id]-d_sxy[id-nz*nx]) - c2*(d_sxy[id+nz*nx]-d_sxy[id-2*nz*nx]))/dy;

			if(gidz<nPml || (gidz>nz-nPml-1)){
				d_mem_dsxz_dz[id] = d_b_z[gidz]*d_mem_dsxz_dz[id] + d_a_z[gidz]*dsxz_dz;
				dsxz_dz = dsxz_dz / d_K_z[gidz] + d_mem_dsxz_dz[id];
			}
			if(gidx<nPml || gidx>nx-nPml){
				d_mem_dsxx_dx[id] = d_b_x_half[gidx]*d_mem_dsxx_dx[id] + d_a_x_half[gidx]*dsxx_dx;
				dsxx_dx = dsxx_dx / d_K_x_half[gidx] + d_mem_dsxx_dx[id];
			}
			if(gidy<nPml || gidy>ny-nPml){
				d_mem_dsxy_dy[id] = d_b_y[gidy]*d_mem_dsxy_dy[id] + d_a_y[gidy]*dsxy_dy;
				dsxy_dy = dsxy_dy / d_K_y[gidy] + d_mem_dsxy_dy[id];
			}

			d_vx[id] += (dsxz_dz + dsxx_dx + dsxy_dy) * d_Byc[id] * dt;

			// update vy
			dsyz_dz = (c1*(d_syz[id]-d_syz[id-1]) - c2*(d_syz[id+1]-d_syz[id-2]))/dz;
			dsxy_dx = (c1*(d_sxy[id]-d_sxy[id-nz]) - c2*(d_sxy[id-nz]-d_sxy[id-2*nz]))/dx;
			dsyy_dy = (c1*(d_syy[id+nz*nx]-d_syy[id]) - c2*(d_syy[id+2*nz*nx]-d_syy[id-nz*nx]))/dy;

			if(gidz<nPml || (gidz>nz-nPml-1)){
				d_mem_dsyz_dz[id] = d_b_z[gidz]*d_mem_dsyz_dz[id] + d_a_z[gidz]*dsyz_dz;
				dsyz_dz = dsyz_dz / d_K_z[gidz] + d_mem_dsyz_dz[id];
			}
			if(gidx<nPml || gidx>nx-nPml){
				d_mem_dsxy_dx[id] = d_b_x[gidx]*d_mem_dsxy_dx[id] + d_a_x[gidx]*dsxy_dx;
				dsxy_dx = dsxy_dx / d_K_x[gidx] + d_mem_dsxy_dx[id];
			}
			if(gidy<nPml || gidy>ny-nPml){
				d_mem_dsyy_dy[id] = d_b_y_half[gidy]*d_mem_dsyy_dy[id] + d_a_y_half[gidy]*dsyy_dy;
				dsyy_dy = dsyy_dy / d_K_y_half[gidy] + d_mem_dsyy_dy[id];
			}

			d_vy[id] += (dsyz_dz + dsxy_dx + dsyy_dy) * d_Byc[id] * dt;
		}
		else{
			return;
		}
	}

	else {

	// ========================================BACKWARD PROPAGATION====================================
	  if(gidz>=nPml && gidz<=nz-1-nPml && gidx>=nPml && gidx<=nx-1-nPml && gidy>=nPml && gidy<=ny-1-nPml) {
		    // update vz
			dszz_dz = (c1*(d_szz[id+1]-d_szz[id]) - c2*(d_szz[id+2]-d_szz[id-1]))/dz;
			dsxz_dx = (c1*(d_sxz[id]-d_sxz[id-nz]) - c2*(d_sxz[id+nz]-d_sxz[id-2*nz]))/dx;
            dsyz_dy = (c1*(d_syz[id]-d_syz[id-nz*nx]) - c2*(d_syz[id+nz*nx]-d_syz[id-2*nz*nx]))/dy;

			d_vz[id] -= (dszz_dz + dsxz_dx + dsyz_dy) * d_Byc[id] * dt;

			// update vx
			dsxz_dz = (c1*(d_sxz[id]-d_sxz[id-1]) - c2*(d_sxz[id+1]-d_sxz[id-2]))/dz;
			dsxx_dx = (c1*(d_sxx[id+nz]-d_sxx[id]) - c2*(d_sxx[id+2*nz]-d_sxx[id-nz]))/dx;
            dsxy_dy = (c1*(d_sxy[id]-d_sxy[id-nz*nx]) - c2*(d_sxy[id+nz*nx]-d_sxy[id-2*nz*nx]))/dy;

			d_vx[id] -= (dsxz_dz + dsxx_dx + dsxy_dy) * d_Byc[id] * dt;

            // update vy
            dsyz_dz = (c1*(d_syz[id]-d_syz[id-1]) - c2*(d_syz[id+1]-d_syz[id-2]))/dz;
			dsxy_dx = (c1*(d_sxy[id]-d_sxy[id-nz]) - c2*(d_sxy[id-nz]-d_sxy[id-2*nz]))/dx;
			dsyy_dy = (c1*(d_syy[id+nz*nx]-d_syy[id]) - c2*(d_syy[id+2*nz*nx]-d_syy[id-nz*nx]))/dy;

            d_vy[id] -= (dsyz_dz + dsxy_dx + dsyy_dy) * d_Byc[id] * dt;

			// computate the density kernel (spray)
			d_DenGrad[id] += -(d_vz_adj[id]*(dszz_dz+dsxz_dx+dsyz_dy)*dt*(-pow(d_Byc[id],2))+\
									     d_vx_adj[id]*(dsxz_dz+dsxx_dx+dsxy_dy)*dt*(-pow(d_Byc[id],2))+\
										 d_vy_adj[id]*(dsyz_dz+dsxy_dx+dsyy_dy)*dt*(-pow(d_Byc[id],2)));
		}
		else{
			return;
		}
	}
}