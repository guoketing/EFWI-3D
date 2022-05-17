#include "Model.h"
#include "Cpml.h"
#include "Parameter.h"
#include "utilities.h"


Cpml::Cpml(Parameter &para, Model &model) {

	int nz = model.nz();
	int nx = model.nx();
	int ny = model.ny();
	int nPml = para.nPoints_pml();
	float f0 = para.f0();
	float dt = para.dt();
	float dz = para.dz();
	float dx = para.dx();
	float dy = para.dy();

	float VpAve = compVpAve(model.h_Vp, nz*nx*ny);

	// for padding
	K_z = (float*)malloc((nz)*sizeof(float));
	a_z = (float*)malloc((nz)*sizeof(float));
	b_z = (float*)malloc((nz)*sizeof(float));
	K_z_half = (float*)malloc((nz)*sizeof(float));
	a_z_half = (float*)malloc((nz)*sizeof(float));
	b_z_half = (float*)malloc((nz)*sizeof(float));

	K_x = (float*)malloc(nx*sizeof(float));
	a_x = (float*)malloc(nx*sizeof(float));
	b_x = (float*)malloc(nx*sizeof(float));
	K_x_half = (float*)malloc(nx*sizeof(float));
	a_x_half = (float*)malloc(nx*sizeof(float));
	b_x_half = (float*)malloc(nx*sizeof(float));

	K_y = (float*)malloc((ny)*sizeof(float));
	a_y = (float*)malloc((ny)*sizeof(float));
	b_y = (float*)malloc((ny)*sizeof(float));
	K_y_half = (float*)malloc((ny)*sizeof(float));
	a_y_half = (float*)malloc((ny)*sizeof(float));
	b_y_half = (float*)malloc((ny)*sizeof(float));

	cpmlInit(K_z, a_z, b_z, K_z_half, \
			a_z_half, b_z_half, nz, nPml, dz, \
			f0, dt, VpAve);

	cpmlInit(K_x, a_x, b_x, K_x_half, \
	        a_x_half, b_x_half, nx, nPml, dx, \
	        f0, dt, VpAve);

	cpmlInit(K_y, a_y, b_y, K_y_half, \
	        a_y_half, b_y_half, ny, nPml, dy, \
	        f0, dt, VpAve);

	// for padding
	CHECK(cudaMalloc((void**)&d_K_z, (nz) *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_a_z, (nz) *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_b_z, (nz) *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_K_z_half, (nz) *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_a_z_half, (nz) *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_b_z_half, (nz) *sizeof(float)));

	CHECK(cudaMalloc((void**)&d_K_x, nx *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_a_x, nx *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_b_x, nx *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_K_x_half, nx *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_a_x_half, nx *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_b_x_half, nx *sizeof(float)));

	CHECK(cudaMalloc((void**)&d_K_y, (ny) *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_a_y, (ny) *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_b_y, (ny) *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_K_y_half, (ny) *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_a_y_half, (ny) *sizeof(float)));
	CHECK(cudaMalloc((void**)&d_b_y_half, (ny) *sizeof(float)));

	// for padding
	CHECK(cudaMemcpy(d_K_z, K_z, (nz)*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_a_z, a_z, (nz)*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_b_z, b_z, (nz)*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_K_z_half, K_z_half, (nz)*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_a_z_half, a_z_half, (nz)*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_b_z_half, b_z_half, (nz)*sizeof(float), cudaMemcpyHostToDevice));

	CHECK(cudaMemcpy(d_K_x, K_x, nx*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_a_x, a_x, nx*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_b_x, b_x, nx*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_K_x_half, K_x_half, nx*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_a_x_half, a_x_half, nx*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_b_x_half, b_x_half, nx*sizeof(float), cudaMemcpyHostToDevice));

	CHECK(cudaMemcpy(d_K_y, K_y, (ny)*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_a_y, a_y, (ny)*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_b_y, b_y, (ny)*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_K_y_half, K_y_half, (ny)*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_a_y_half, a_y_half, (ny)*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_b_y_half, b_y_half, (ny)*sizeof(float), cudaMemcpyHostToDevice));

}


Cpml::~Cpml() {
	free(K_z);
	free(a_z);
	free(b_z);
	free(K_z_half);
	free(a_z_half);
	free(b_z_half);
	free(K_x);
	free(a_x);
	free(b_x);
	free(K_x_half);
	free(a_x_half);
	free(b_x_half);
	free(K_y);
	free(a_y);
	free(b_y);
	free(K_y_half);
	free(a_y_half);
	free(b_y_half);

	CHECK(cudaFree(d_K_z));
	CHECK(cudaFree(d_a_z));
	CHECK(cudaFree(d_b_z));
	CHECK(cudaFree(d_K_z_half));
	CHECK(cudaFree(d_a_z_half));
	CHECK(cudaFree(d_b_z_half));
	CHECK(cudaFree(d_K_x));
	CHECK(cudaFree(d_a_x));
	CHECK(cudaFree(d_b_x));
	CHECK(cudaFree(d_K_x_half));
	CHECK(cudaFree(d_a_x_half));
	CHECK(cudaFree(d_b_x_half));
	CHECK(cudaFree(d_K_y));
	CHECK(cudaFree(d_a_y));
	CHECK(cudaFree(d_b_y));
	CHECK(cudaFree(d_K_y_half));
	CHECK(cudaFree(d_a_y_half));
	CHECK(cudaFree(d_b_y_half));
}