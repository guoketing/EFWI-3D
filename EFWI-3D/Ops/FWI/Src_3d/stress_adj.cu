__global__ void stress_adj(float *d_vz, float *d_vx, float *d_vy, float *d_szz, float *d_sxx, float *d_sxz, \
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

	float dphi_xx_x_dx = 0.0;
	float dvx_dx = 0.0;
	float dphi_zz_z_dz = 0.0;
	float dvz_dz = 0.0;
	float dphi_yy_y_dy = 0.0;
	float dvy_dy = 0.0;

	float dphi_xy_y_dy = 0.0;
	float dvx_dy = 0.0;
	float dphi_xy_x_dx = 0.0;
	float dvy_dx = 0.0;

	float dphi_xz_z_dz = 0.0;
	float dvx_dz = 0.0;
	float dphi_xz_x_dx = 0.0;
	float dvz_dx = 0.0;

	float dphi_yz_z_dz = 0.0;
	float dvy_dz = 0.0;
    float dphi_yz_y_dy = 0.0;
	float dvz_dy = 0.0;

    float c1 = 9.0/8.0;
	float c2 = 1.0/24.0;
	
	float lambda = d_Lambda[id];
    float mu = d_Mu[id];
	float Byc = d_Byc[id];


	if (gidz>=2 && gidz<=nz-3 && gidx>=2 && gidx<=nx-3  && gidy>=2 && gidy<=ny-3) {

		dphi_xz_x_dx = (-c1*(d_mem_dsxz_dx[id+nz]-d_mem_dsxz_dx[id]) \
				+ c2*(d_mem_dsxz_dx[id+2*nz]-d_mem_dsxz_dx[id-nz]))/dx;
		dvz_dx = (-c1*(d_vz[id+nz]-d_vz[id]) + c2*(d_vz[id+2*nz]-d_vz[id-nz]))/dx;
		dphi_xz_z_dz = (-c1*(d_mem_dsxz_dz[id+1]-d_mem_dsxz_dz[id]) \
				+ c2*(d_mem_dsxz_dz[id+2]-d_mem_dsxz_dz[id-1]))/dz;
		dvx_dz = (-c1*(d_vx[id+1]-d_vx[id]) + c2*(d_vx[id+2]-d_vx[id-1]))/dz;

		// update sxz
		d_sxz[id] += d_a_x[gidx]*dphi_xz_x_dx + dvz_dx/d_K_x[gidx]*Byc*dt \
				+ d_a_z[gidz]*dphi_xz_z_dz + dvx_dz/d_K_z[gidz]*Byc*dt;

		// update psi_zx and psi_xz

		if(gidx<nPml || gidx>nx-nPml-1){
			d_mem_dvz_dx[id] = d_b_x_half[gidx]*d_mem_dvz_dx[id] + d_sxz[id]*mu*dt;
		}
		if(gidz<nPml || gidz>nz-nPml-1){
			d_mem_dvx_dz[id] = d_b_z_half[gidz]*d_mem_dvx_dz[id] + d_sxz[id]*mu*dt;
		}

		dphi_xy_x_dx = (-c1*(d_mem_dsxy_dx[id+nz]-d_mem_dsxy_dx[id]) \
				+ c2*(d_mem_dsxy_dx[id+2*nz]-d_mem_dsxy_dx[id-nz]))/dx;
		dvy_dx = (-c1*(d_vy[id+nz]-d_vy[id]) + c2*(d_vy[id+2*nz]-d_vy[id-nz]))/dx;
		dphi_xy_y_dy = (-c1*(d_mem_dsxy_dy[id+nz*nx]-d_mem_dsxy_dy[id]) \
				+ c2*(d_mem_dsxy_dy[id+2*nz*nx]-d_mem_dsxy_dy[id-nz*nx]))/dy;
		dvx_dy = (-c1*(d_vx[id+nz*nx]-d_vx[id]) + c2*(d_vx[id+2*nz*nx]-d_vx[id-nz*nx]))/dy;

		// update sxy
		d_sxy[id] += d_a_x[gidx]*dphi_xy_x_dx + dvy_dx/d_K_x[gidx]*Byc*dt \
				+ d_a_y[gidy]*dphi_xy_y_dy + dvx_dy/d_K_y[gidy]*Byc *dt;

		// update psi_zx and psi_xz
		if(gidy<nPml || gidy>ny-nPml-1){
			d_mem_dvx_dy[id] = d_b_y_half[gidy]*d_mem_dvx_dy[id] + d_sxy[id]*mu*dt;
		}
		if(gidx<nPml || gidx>nx-nPml-1){
			d_mem_dvy_dx[id] = d_b_x_half[gidx]*d_mem_dvy_dx[id] + d_sxy[id]*mu*dt;
		}

		dphi_yz_y_dy = (-c1*(d_mem_dsyz_dy[id+nz*nx]-d_mem_dsyz_dy[id]) \
				+ c2*(d_mem_dsyz_dy[id+2*nz*nx]-d_mem_dsyz_dy[id-nz*nx]))/dy;
		dvz_dy = (-c1*(d_vz[id+nz*nx]-d_vz[id]) + c2*(d_vz[id+2*nz*nx]-d_vz[id-nz*nx]))/dy;		
		dphi_yz_z_dz = (-c1*(d_mem_dsyz_dz[id+1]-d_mem_dsyz_dz[id]) \
				+ c2*(d_mem_dsyz_dz[id+2]-d_mem_dsyz_dz[id-1]))/dz;
		dvy_dz = (-c1*(d_vy[id+1]-d_vy[id]) + c2*(d_vy[id+2]-d_vy[id-1]))/dz;

		// update syz
		d_syz[id] += d_a_y[gidy]*dphi_yz_y_dy + dvz_dy/d_K_y[gidy]*Byc*dt \
				+ d_a_z[gidz]*dphi_yz_z_dz + dvy_dz/d_K_z[gidz]*Byc*dt;

		// update psi_zx and psi_xz
		if(gidy<nPml || gidy>ny-nPml-1){
			d_mem_dvz_dy[id] = d_b_y_half[gidy]*d_mem_dvz_dy[id] + d_syz[id]*mu*dt;
		}
		if(gidz<nPml || gidz>nz-nPml-1){
			d_mem_dvy_dz[id] = d_b_z_half[gidz]*d_mem_dvy_dz[id] + d_syz[id]*mu*dt;
		}
		
		dphi_xx_x_dx = (-c1*(d_mem_dsxx_dx[id]-d_mem_dsxx_dx[id-nz]) \
				+ c2*(d_mem_dsxx_dx[id+nz]-d_mem_dsxx_dx[id-2*nz]))/dx;
		dvx_dx = (-c1*(d_vx[id]-d_vx[id-nz]) + c2*(d_vx[id+nz]-d_vx[id-2*nz]))/dx;
		dphi_zz_z_dz = (-c1*(d_mem_dszz_dz[id]-d_mem_dszz_dz[id-1]) \
				+ c2*(d_mem_dszz_dz[id+1]-d_mem_dszz_dz[id-2]))/dz;
		dvz_dz = (-c1*(d_vz[id]-d_vz[id-1]) + c2*(d_vz[id+1]-d_vz[id-2]))/dz;
		dphi_yy_y_dy = (-c1*(d_mem_dsyy_dy[id]-d_mem_dsyy_dy[id-nz*nx]) \
				+ c2*(d_mem_dsyy_dy[id+nz*nx]-d_mem_dsyy_dy[id-2*nz*nx]))/dy;
		dvy_dy = (-c1*(d_vy[id]-d_vy[id-nz*nx]) + c2*(d_vy[id+nz*nx]-d_vy[id-2*nz*nx]))/dy;


		// update sxx and szz
		d_sxx[id] += d_a_x_half[gidx]*dphi_xx_x_dx + d_Byc[id]*dvx_dx/d_K_x_half[gidx]*dt;;
		d_szz[id] += d_a_z_half[gidz]*dphi_zz_z_dz + d_Byc[id]*dvz_dz/d_K_z_half[gidz]*dt;
		d_syy[id] += d_a_y_half[gidy]*dphi_yy_y_dy + d_Byc[id]*dvy_dy/d_K_y_half[gidy]*dt;


		// update psi_xx and psi_zz
		if(gidx<nPml || gidx>nx-nPml-1){
		d_mem_dvx_dx[id] = d_b_x[gidx]*d_mem_dvx_dx[id] + lambda*d_szz[id]*dt \
				+ (lambda+2.0*mu)*d_sxx[id]*dt + lambda*d_syy[id]*dt;
		}
		if(gidz<nPml || gidz>nz-nPml-1){
		d_mem_dvz_dz[id] = d_b_z[gidz]*d_mem_dvz_dz[id] + (lambda+2.0*mu)*d_szz[id]*dt \
				+ lambda*d_sxx[id]*dt + lambda*d_syy[id]*dt;
		}
		if(gidy<nPml || gidy>ny-nPml-1){
		d_mem_dvy_dy[id] = d_b_y[gidy]*d_mem_dvy_dy[id] + lambda*d_szz[id]*dt \
				+ lambda*d_sxx[id]*dt + (lambda+2.0*mu)*d_syy[id]*dt;
		}
	}
	else {
		return;
	}
}
