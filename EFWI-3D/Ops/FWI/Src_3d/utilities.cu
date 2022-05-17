#define d_Mu(z,x,y) d_Mu[(y)*(nx*nz)+(x)*(nz)+(z)]
// #define d_ave_Mu(x, y, z) d_ave_Mu[(z) * (nx * ny) + (y) * (nx) + (x)]
#define d_field(z, x, y) d_field[y * (nx * nz) + (x) * (nz) + (z)]
#define d_bnd(x, indT) d_bnd[(indT) * (len_Bnd_vec) + (x)]
#define d_Rho(z,x,y) d_Rho[(y)*(nx*nz)+(x)*(nz)+(z)]
#define d_Byc(z,x,y) d_Byc[(y)*(nx*nz)+(x)*(nz)+(z)]
#include "utilities.h"

void fileBinLoad(float *h_bin, int size, std::string fname) {
  FILE *fp = fopen(fname.c_str(), "rb");
  if (fp == nullptr) {
    std::cout << "Attempted to read " << fname << std::endl;
    printf("File reading error!\n");
    exit(1);
  } else {
    size_t sizeRead = fread(h_bin, sizeof(float), size, fp);
  }
  fclose(fp);
}

void fileBinWrite(float *h_bin, int size, std::string fname) {
  FILE *fp = fopen(fname.c_str(), "wb");
  if (fp == nullptr) {
    printf("File writing error!\n");
    exit(1);
  } else {
    fwrite(h_bin, sizeof(float), size, fp);
  }
  fclose(fp);
}

void fileBinWriteDouble(double *h_bin, int size, std::string fname) {
  FILE *fp = fopen(fname.c_str(), "wb");
  if (fp == nullptr) {
    printf("File writing error!\n");
    exit(1);
  } else {
    fwrite(h_bin, sizeof(double), size, fp);
  }
  fclose(fp);
}

void initialArray(float *ip, int size, float value) {
  for (int i = 0; i < size; i++) {
    ip[i] = value;
    // printf("value = %f\n", value);
  }
}

void initialArray(double *ip, int size, double value) {
  for (int i = 0; i < size; i++) {
    ip[i] = value;
    // printf("value = %f\n", value);
  }
}

__global__ void initialArrayGPU(float *ip, int nz, int nx, int ny, float value) {
  int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
  
  if (threadId < (nx * ny * nz)) {
      ip[threadId] = value;
  }
}


__global__ void initial2DArrayGPU(float *ip, int nx, int ny, float value) {
  int gidx = threadIdx.x + blockDim.x * blockIdx.x;
  int gidy = threadIdx.y + blockDim.y * blockIdx.y;
  if (gidx < nx && gidy < ny) {
    int offset = gidy * nx + gidx;
    ip[offset] = value;
  }
}

__global__ void assignArrayGPU(float *ip_in, float *ip_out, int nx, int ny) {
  int gidx = threadIdx.x + blockDim.x * blockIdx.x;
  int gidy = threadIdx.y + blockDim.y * blockIdx.y;
  if (gidx < nx && gidy < ny) {
    int offset = gidx + gidy * nx;
    ip_out[offset] = ip_in[offset];
  }
}

void displayArray(std::string s, float *ip, int nx, int ny) {
  // printf("ip: \n");
  // printf("%s: \n", s);
  std::cout << s << ": " << std::endl;
  for (int i = 0; i < ny; i++) {
    for (int j = 0; j < nx; j++) {
      // printf("ip[%d, %d] = %f  ", i, j, ip[i*nx+j]);
      printf("%f  ", ip[i * nx + j]);
    }
    printf("\n");
  }
  printf("\n\n\n");
}


__global__ void velInit(float *d_Lambda, float *d_Mu, float *d_Rho, float *d_Vp,
                        float *d_Vs, int nz, int nx, int ny) {
  // printf("Hello, world!\n");
  int gidx = threadIdx.x + blockDim.x * blockIdx.x;
  int gidy = threadIdx.y + blockDim.y * blockIdx.y;
  int gidz = threadIdx.z + blockDim.z * blockIdx.z;
  int offset = gidy * (nx * nz) +  gidx * nz + gidz;
  if (gidx < nz && gidy < nx && gidz < ny) {
    d_Vp[offset] =
        sqrt((d_Lambda[offset] + 2.0 * d_Mu[offset]) / d_Rho[offset]);
    d_Vs[offset] = sqrt((d_Mu[offset]) / d_Rho[offset]);
  }
}


__global__ void aveBycInit(float *d_Rho, float *d_Byc, int nz, int nx, int ny) {
  int gidx = threadIdx.x + blockDim.x * blockIdx.x;
  int gidy = threadIdx.y + blockDim.y * blockIdx.y;
  int gidz = threadIdx.z + blockDim.z * blockIdx.z;

  int offset = gidy * (nx * nz) +  gidx * nz + gidz;
  if (gidz >= 2 && gidz <= nz - 3 && gidx >= 2 && gidx <= nx - 3 && gidy >= 2 && gidy <= ny - 3) {
    d_Byc[offset] = 1.0 / d_Rho[offset];
  } else {
    return;
  }
}

__global__ void gpuMinus(float *d_out, float *d_in1, float *d_in2, int nx,
                         int ny) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.y + blockDim.y * blockIdx.y;
  // only compute last N-1 time samples for misfits!!!!!!!! DL 02/25/2019
  if (idx < nx && idy < ny && idx > 0) {
    d_out[(idy) * (nx) + (idx)] =
        d_in1[(idy) * (nx) + (idx)] - d_in2[(idy) * (nx) + (idx)];
  } else if (idx == 0 && idy < ny) {
    d_out[(idy) * (nx) + (idx)] = 0.0;
  } else {
    return;
  }
}

__global__ void cuda_cal_objective(float *obj, float *err, int ng){
/*< calculate the value of objective function: obj >*/
  const int Block_Size = 512;
  __shared__ float sdata[Block_Size];
  int tid = threadIdx.x;
  sdata[tid] = 0.0f;
  for (int s = 0; s < (ng + Block_Size - 1) / Block_Size; s++) {
    int id = s * blockDim.x + threadIdx.x;
    float a = (id < ng) ? err[id] : 0.0f;
    sdata[tid] += powf(a, 2);
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s >= 1; s /= 2) {
    if (threadIdx.x < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0) {
    *obj = sdata[0];
  }
}

float cal_objective(float *array, int N) {
  float misfit = 0.0;
  printf("hhh\n");
  for (int i = 0; i < N; i++) {
    misfit += array[i] * array[i];
  }
  return misfit;
}

float compVpAve(float *array, int N) {
  float temp = 0.0;
  for (int i = 0; i < N; i++) {
    temp += array[i];
  }
  temp = temp / float(N);
  return temp;
}

void compCourantNumber(float *h_Vp, int size, float dt, float dz, float dx, float dy) {
  float max = h_Vp[0];
  float Courant_number = 0.0;
  for (int i = 0; i < size; i++) {
    if (h_Vp[i] > max) {
      max = h_Vp[i];
    }
  }
  float dh_min = (dz < dx) ? dz : dx;
  // Courant_number = max * dt * sqrtf(powf(1.0 / dz, 2) + powf(1.0 / dx, 2));
  Courant_number = max * dt * sqrtf(2.0) * (1.0 / 24.0 + 9.0 / 8.0) / dh_min;
  
  if (Courant_number > 1.0) {
    std::cout << "Courant_number = " << Courant_number << std::endl;
    exit(1);
  }
}

void cpmlInit(float *K, float *a, float *b, float *K_half, float *a_half,
              float *b_half, int N, int nPml, float dh, float f0, float dt,
              float VpAve) {
  float *damp, *damp_half, *alpha, *alpha_half;
  float d0_h = 0.0;
  float Rcoef = 0.0008;
  float depth_in_pml = 0.0;
  float depth_normalized = 0.0;
  float thickness_PML = 0.0;
  // const float PI = 3.141592653589793238462643383279502884197169;
  const float K_MAX_PML = 2.0;
  const float ALPHA_MAX_PML = 2.0 * PI * (f0 / 2.0);
  const float NPOWER = 8.0;
  const float c1 = 0.25, c2 = 0.75, c3 = 0.0;
  // const float c1 = 0.0, c2 = 1.0, c3 = 0.0;

  thickness_PML = nPml * dh;  // changed here
  VpAve = 3000.0;             // DL make this model independent
  d0_h = -(NPOWER + 1) * VpAve * log(Rcoef) / (2.0 * thickness_PML);
  damp = (float *)malloc(N * sizeof(float));
  damp_half = (float *)malloc(N * sizeof(float));
  alpha = (float *)malloc(N * sizeof(float));
  alpha_half = (float *)malloc(N * sizeof(float));
  initialArray(damp, N, 0.0);
  initialArray(damp_half, N, 0.0);
  initialArray(K, N, 1.0);
  initialArray(K_half, N, 1.0);
  initialArray(alpha, N, 0.0);
  initialArray(alpha_half, N, 0.0);
  initialArray(a, N, 0.0);
  initialArray(a_half, N, 0.0);
  initialArray(b, N, 0.0);
  initialArray(b_half, N, 0.0);

  for (int i = 0; i < N; i++) {
    // left edge
    depth_in_pml = (nPml - i) * dh;
    if (depth_in_pml >= 0.0) {
      depth_normalized = depth_in_pml / thickness_PML;
      damp[i] =
          d0_h * (c1 * depth_normalized + c2 * pow(depth_normalized, NPOWER) +
                  c3 * pow(depth_normalized, 2 * NPOWER));
      K[i] = 1.0 + (K_MAX_PML - 1.0) * pow(depth_normalized, NPOWER);
      alpha[i] = ALPHA_MAX_PML * (1.0 - depth_normalized);
    }
    if (alpha[i] < 0.0) {
      std::cout << "CPML alpha < 0.0 --" << __LINE__ << std::endl;
      exit(1);
    }

    // half the grid points
    depth_in_pml = (nPml - i - 0.5) * dh;
    if (depth_in_pml >= 0.0) {
      depth_normalized = depth_in_pml / thickness_PML;
      damp_half[i] =
          d0_h * (c1 * depth_normalized + c2 * pow(depth_normalized, NPOWER) +
                  c3 * pow(depth_normalized, 2 * NPOWER));
      K_half[i] = 1.0 + (K_MAX_PML - 1.0) * pow(depth_normalized, NPOWER);
      alpha_half[i] = ALPHA_MAX_PML * (1.0 - depth_normalized);
    }
    if (alpha_half[i] < 0.0) {
      std::cout << "CPML alpha_half < 0.0 --" << __LINE__ << std::endl;
      exit(1);
    }

    // right edge
    depth_in_pml = (nPml - N + i) * dh;
    if (depth_in_pml >= 0.0) {
      depth_normalized = depth_in_pml / thickness_PML;
      damp[i] =
          d0_h * (c1 * depth_normalized + c2 * pow(depth_normalized, NPOWER) +
                  c3 * pow(depth_normalized, 2 * NPOWER));
      K[i] = 1.0 + (K_MAX_PML - 1.0) * pow(depth_normalized, NPOWER);
      alpha[i] = ALPHA_MAX_PML * (1.0 - depth_normalized);
    }
    if (alpha[i] < 0.0) {
      std::cout << "CPML alpha < 0.0 --" << __LINE__ << std::endl;
      exit(1);
    }

    depth_in_pml = (nPml - N + i + 0.5) * dh;
    if (depth_in_pml >= 0.0) {
      depth_normalized = depth_in_pml / thickness_PML;
      damp_half[i] =
          d0_h * (c1 * depth_normalized + c2 * pow(depth_normalized, NPOWER) +
                  c3 * pow(depth_normalized, 2 * NPOWER));
      K_half[i] = 1.0 + (K_MAX_PML - 1.0) * powf(depth_normalized, NPOWER);
      alpha_half[i] = ALPHA_MAX_PML * (1.0 - depth_normalized);
    }
    if (alpha_half[i] < 0.0) {
      std::cout << "CPML alpha_half < 0.0 --" << __LINE__ << std::endl;
      exit(1);
    }

    if (alpha[i] < 0.0) {
      alpha[i] = 0.0;
    }
    if (alpha_half[i] < 0.0) {
      alpha_half[i] = 0.0;
    }

    b[i] = expf(-(damp[i] / K[i] + alpha[i]) * dt);
    b_half[i] = expf(-(damp_half[i] / K_half[i] + alpha_half[i]) * dt);

    if (fabs(damp[i]) > 1.0e-6) {
      a[i] = damp[i] * (b[i] - 1.0) / (K[i] * (damp[i] + K[i] * alpha[i]));
    }
    if (fabs(damp_half[i]) > 1.0e-6) {
      a_half[i] = damp_half[i] * (b_half[i] - 1.0) /
                  (K_half[i] * (damp_half[i] + K_half[i] * alpha_half[i]));
    }
  }
  free(damp);
  free(damp_half);
  free(alpha);
  free(alpha_half);
}

// Keting Guo 02/17/2022
__global__ void from_bnd(float *d_field, float *d_bnd, int nz, int nx, int ny,
                         int nzBnd, int nxBnd, int nyBnd, int len_Bnd_vec, int nLayerStore,
                         int indT, int nPml, int nSteps) {
  int idxBnd = threadIdx.x + blockDim.x * blockIdx.x;
  int iRow, jCol, kLayer;

  if (idxBnd >= 0 && idxBnd <= nLayerStore * (nzBnd * nxBnd) - 1) {
    kLayer = idxBnd / (nzBnd * nxBnd);
    jCol = (idxBnd - kLayer * (nzBnd * nxBnd)) / nzBnd;
    iRow = (idxBnd - kLayer * (nzBnd * nxBnd)) - jCol * nzBnd;
    d_bnd(idxBnd, indT) = d_field((iRow + nPml - 2), (jCol + nPml - 2), (kLayer + nPml -2));
  } else if (idxBnd >= nLayerStore * (nzBnd * nxBnd) &&
             idxBnd <= 2 * nLayerStore * (nzBnd * nxBnd) - 1) {
    kLayer = (idxBnd - nLayerStore * (nzBnd * nxBnd)) / (nzBnd * nxBnd);
    jCol = ((idxBnd - nLayerStore * (nzBnd * nxBnd)) - kLayer * (nzBnd * nxBnd)) / nzBnd;
    iRow = ((idxBnd - nLayerStore * (nzBnd * nxBnd)) - kLayer * (nzBnd * nxBnd)) - jCol * nzBnd;
    d_bnd(idxBnd, indT) =
        d_field((iRow + nPml - 2), (jCol + nPml - 2), (ny - nPml - kLayer - 1 + 2));
  } else if (idxBnd >= 2 * nLayerStore * (nzBnd * nxBnd) &&
             idxBnd <= nLayerStore * (2 * (nzBnd * nxBnd) + (nxBnd * nyBnd)) - 1) {
    iRow = (idxBnd - 2 * nLayerStore * (nzBnd * nxBnd)) / (nxBnd * nyBnd);
    kLayer = ((idxBnd - 2 * nLayerStore * (nzBnd * nxBnd)) - iRow * (nxBnd * nyBnd)) / nxBnd;
    jCol = ((idxBnd - 2 * nLayerStore * (nzBnd * nxBnd)) - iRow * (nxBnd * nyBnd)) - kLayer * nxBnd;
    d_bnd(idxBnd, indT) = d_field((iRow + nPml - 2), (jCol + nPml - 2), (kLayer + nPml -2));
  } else if (idxBnd >= nLayerStore * (2 * (nzBnd * nxBnd)+ (nxBnd * nyBnd)) &&
             idxBnd <= 2 * nLayerStore * ((nzBnd * nxBnd) + (nxBnd * nyBnd)) - 1) {
    iRow = (idxBnd - nLayerStore * (2 * (nzBnd * nxBnd) + (nxBnd * nyBnd))) / (nxBnd * nyBnd);
    kLayer = ((idxBnd - nLayerStore * (2 * (nzBnd * nxBnd) + (nxBnd * nyBnd))) - iRow * (nxBnd * nyBnd)) / nxBnd;
    jCol = ((idxBnd - nLayerStore * (2 * (nzBnd * nxBnd) + (nxBnd * nyBnd))) - iRow * (nxBnd * nyBnd)) - kLayer * nxBnd;
    d_bnd(idxBnd, indT) =
        d_field((nz - nPml - iRow - 1 + 2), (jCol + nPml - 2), (kLayer + nPml -2));
  } else if (idxBnd >= nLayerStore * 2 * ((nzBnd * nxBnd)+ (nxBnd * nyBnd)) &&
             idxBnd <= nLayerStore * (2 * ((nzBnd * nxBnd) + (nxBnd * nyBnd)) + (nyBnd * nzBnd)) - 1) {
    jCol = (idxBnd - nLayerStore * 2 * ((nzBnd * nxBnd)+ (nxBnd * nyBnd))) / (nyBnd * nzBnd);
    iRow = ((idxBnd - nLayerStore * 2 * ((nzBnd * nxBnd)+ (nxBnd * nyBnd))) - jCol * (nyBnd * nzBnd)) / nyBnd;
    kLayer = ((idxBnd - nLayerStore * 2 * ((nzBnd * nxBnd)+ (nxBnd * nyBnd))) - jCol * (nyBnd * nzBnd)) - iRow * nyBnd;
    d_bnd(idxBnd, indT) =
        d_field((iRow + nPml - 2), (jCol + nPml - 2), (kLayer + nPml -2));
  } else if (idxBnd >= nLayerStore * (2 * ((nzBnd * nxBnd) + (nxBnd * nyBnd)) + (nyBnd * nzBnd)) &&
             idxBnd <= nLayerStore * 2 * ((nzBnd * nxBnd) + (nxBnd * nyBnd)+ (nyBnd * nzBnd)) - 1) {
    jCol = (idxBnd - nLayerStore * (2 * ((nzBnd * nxBnd) + (nxBnd * nyBnd)) + (nyBnd * nzBnd))) / (nyBnd * nzBnd);
    iRow = ((idxBnd - nLayerStore * (2 * ((nzBnd * nxBnd) + (nxBnd * nyBnd)) + (nyBnd * nzBnd))) - jCol * (nyBnd * nzBnd)) / nyBnd;
    kLayer = ((idxBnd - nLayerStore * (2 * ((nzBnd * nxBnd) + (nxBnd * nyBnd)) + (nyBnd * nzBnd))) - jCol * (nyBnd * nzBnd)) - iRow * nyBnd;
    d_bnd(idxBnd, indT) =
        d_field((iRow + nPml - 2), (nx - nPml - jCol - 1 + 2), (kLayer + nPml -2));
  }
   else {
    return;
  }
}


__global__ void to_bnd(float *d_field, float *d_bnd, int nz, int nx, int ny, int nzBnd,
                       int nxBnd, int nyBnd, int len_Bnd_vec, int nLayerStore, int indT,
                       int nPml, int nSteps) {
  int idxBnd = threadIdx.x + blockDim.x * blockIdx.x;
  int iRow, jCol, kLayer;

  if (idxBnd >= 0 && idxBnd <= nLayerStore * (nzBnd * nxBnd) - 1) {
    kLayer = idxBnd / (nzBnd * nxBnd);
    jCol = (idxBnd - kLayer * (nzBnd * nxBnd)) / nzBnd;
    iRow = (idxBnd - kLayer * (nzBnd * nxBnd)) - jCol * nzBnd;
    d_field((iRow + nPml - 2), (jCol + nPml - 2), (kLayer + nPml -2)) = d_bnd(idxBnd, indT);
  } else if (idxBnd >= nLayerStore * (nzBnd * nxBnd) &&
             idxBnd <= 2 * nLayerStore * (nzBnd * nxBnd) - 1) {
    kLayer = (idxBnd - nLayerStore * (nzBnd * nxBnd)) / (nzBnd * nxBnd);
    jCol = ((idxBnd - nLayerStore * (nzBnd * nxBnd)) - kLayer * (nzBnd * nxBnd)) / nzBnd;
    iRow = ((idxBnd - nLayerStore * (nzBnd * nxBnd)) - kLayer * (nzBnd * nxBnd)) - jCol * nzBnd;
    d_field((iRow + nPml - 2), (jCol + nPml - 2), (ny - nPml - kLayer - 1 + 2)) =  d_bnd(idxBnd, indT);
  } else if (idxBnd >= 2 * nLayerStore * (nzBnd * nxBnd) &&
             idxBnd <= nLayerStore * (2 * (nzBnd * nxBnd) + (nxBnd * nyBnd)) - 1) {
    iRow = (idxBnd - 2 * nLayerStore * (nzBnd * nxBnd)) / (nxBnd * nyBnd);
    kLayer = ((idxBnd - 2 * nLayerStore * (nzBnd * nxBnd)) - iRow * (nxBnd * nyBnd)) / nxBnd;
    jCol = ((idxBnd - 2 * nLayerStore * (nzBnd * nxBnd)) - iRow * (nxBnd * nyBnd)) - kLayer * nxBnd;
    d_field((iRow + nPml - 2), (jCol + nPml - 2), (kLayer + nPml -2)) = d_bnd(idxBnd, indT);
  } else if (idxBnd >= nLayerStore * (2 * (nzBnd * nxBnd)+ (nxBnd * nyBnd)) &&
             idxBnd <= 2 * nLayerStore * ((nzBnd * nxBnd) + (nxBnd * nyBnd)) - 1) {
    iRow = (idxBnd - nLayerStore * (2 * (nzBnd * nxBnd) + (nxBnd * nyBnd))) / (nxBnd * nyBnd);
    kLayer = ((idxBnd - nLayerStore * (2 * (nzBnd * nxBnd) + (nxBnd * nyBnd))) - iRow * (nxBnd * nyBnd)) / nxBnd;
    jCol = ((idxBnd - nLayerStore * (2 * (nzBnd * nxBnd) + (nxBnd * nyBnd))) - iRow * (nxBnd * nyBnd)) - kLayer * nxBnd;
    d_field((nz - nPml - iRow - 1 + 2), (jCol + nPml - 2), (kLayer + nPml -2)) =  d_bnd(idxBnd, indT);
  } else if (idxBnd >= nLayerStore * 2 * ((nzBnd * nxBnd)+ (nxBnd * nyBnd)) &&
             idxBnd <= nLayerStore * (2 * ((nzBnd * nxBnd) + (nxBnd * nyBnd)) + (nyBnd * nzBnd)) - 1) {
    jCol = (idxBnd - nLayerStore * 2 * ((nzBnd * nxBnd)+ (nxBnd * nyBnd))) / (nyBnd * nzBnd);
    iRow = ((idxBnd - nLayerStore * 2 * ((nzBnd * nxBnd)+ (nxBnd * nyBnd))) - jCol * (nyBnd * nzBnd)) / nyBnd;
    kLayer = ((idxBnd - nLayerStore * 2 * ((nzBnd * nxBnd)+ (nxBnd * nyBnd))) - jCol * (nyBnd * nzBnd)) - iRow * nyBnd;
    d_field((iRow + nPml - 2), (jCol + nPml - 2), (kLayer + nPml -2)) = d_bnd(idxBnd, indT);
  } else if (idxBnd >= nLayerStore * (2 * ((nzBnd * nxBnd) + (nxBnd * nyBnd)) + (nyBnd * nzBnd)) &&
             idxBnd <= nLayerStore * 2 * ((nzBnd * nxBnd) + (nxBnd * nyBnd)+ (nyBnd * nzBnd)) - 1) {
    jCol = (idxBnd - nLayerStore * (2 * ((nzBnd * nxBnd) + (nxBnd * nyBnd)) + (nyBnd * nzBnd))) / (nyBnd * nzBnd);
    iRow = ((idxBnd - nLayerStore * (2 * ((nzBnd * nxBnd) + (nxBnd * nyBnd)) + (nyBnd * nzBnd))) - jCol * (nyBnd * nzBnd)) / nyBnd;
    kLayer = ((idxBnd - nLayerStore * (2 * ((nzBnd * nxBnd) + (nxBnd * nyBnd)) + (nyBnd * nzBnd))) - jCol * (nyBnd * nzBnd)) - iRow * nyBnd;
    d_field((iRow + nPml - 2), (nx - nPml - jCol - 1 + 2), (kLayer + nPml -2)) = d_bnd(idxBnd, indT);
  }
   else {
    return;
  }
}


__global__ void src_rec_gauss_amp(float *gauss_amp, int nz, int nx, int ny) {
  int gidz = blockIdx.x * blockDim.x + threadIdx.x;
  int gidx = blockIdx.y * blockDim.y + threadIdx.y;
  int gidy = blockIdx.z * blockDim.z + threadIdx.z;
  if (gidz >= 0 && gidz < nz && gidx >= 0 && gidx < nx && gidy >= 0 && gidy < ny) {
    int idz = gidz - nz / 2;
    int idx = gidx - nx / 2;
    int idy = gidy - ny / 2;
    gauss_amp[gidz + gidx * nz + gidy * (nx * nz)] =
        expf(-1000.0 * (powf(float(idz), 2) + powf(float(idx), 2) + powf(float(idy), 2)));
    // printf("gidz=%d, gidx=%d, gidy=%d, gauss_amp=%.10f\n", gidz, gidx, gidy,
    //        gauss_amp[gidz + gidx * nz + gidy * (nx * ny)]);
  } else {
    return;
  }
}



__global__ void add_source(float *d_szz, float *d_sxx, float *d_syy, float amp, int nz, int nx, int ny,
                           bool isFor, int z_loc, int x_loc, int y_loc, float dt,
                           float *gauss_amp) {
  // int id = threadIdx.x + blockDim.x * blockIdx.x;
  int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int gidy = blockIdx.y * blockDim.y + threadIdx.y;
  int gidz = blockIdx.z * blockDim.z + threadIdx.z;

  float scale = pow(1500.0, 2);
  if (isFor) {
    if (gidz >= 0 && gidz < 9 && gidx >= 0 && gidx < 9 && gidy >= 0 && gidy < 9) {
      int idz = gidz - 9 / 2;
      int idx = gidx - 9 / 2;
      int idy = gidy - 9 / 2;

      
      d_szz[(y_loc+idy) * (nx*nz) + (x_loc+idx) * nz + (z_loc+idz)] += scale * amp * dt * gauss_amp[gidz + gidx * 9 + gidy * 81];
      // crosswell borehole source (can be modified) assume cp/cs = sqrt(3.0)
      d_sxx[(y_loc+idy) * (nx*nz) + (x_loc+idx) * nz + (z_loc+idz)] += scale * amp * dt * gauss_amp[gidz + gidx * 9 + gidy * 81];
      d_syy[(y_loc+idy) * (nx*nz) + (x_loc+idx) * nz + (z_loc+idz)] += scale * amp * dt * gauss_amp[gidz + gidx * 9 + gidy * 81];
    } else {
      return;
    }
  } else {
    if (gidz >= 0 && gidz < 9 && gidx >= 0 && gidx < 9 && gidy >= 0 && gidy < 9) {
      int idz = gidz - 9 / 2;
      int idx = gidx - 9 / 2;
      int idy = gidy - 9 / 2;
      // printf("amp = %f  ", amp);
      d_szz[(y_loc+idy) * (nx*nz) + (x_loc+idx) * nz + (z_loc+idz)] -= scale * amp * dt * gauss_amp[gidz + gidx * 9 + gidy * 81];
      // crosswell borehole source (can be modified) assume cp/cs = sqrt(3.0)
      d_sxx[(y_loc+idy) * (nx*nz) + (x_loc+idx) * nz + (z_loc+idz)] -= scale * amp * dt * gauss_amp[gidz + gidx * 9 + gidy * 81];
      d_syy[(y_loc+idy) * (nx*nz) + (x_loc+idx) * nz + (z_loc+idz)] -= scale * amp * dt * gauss_amp[gidz + gidx * 9 + gidy * 81];
    } else {
      return;
    }
  }
}

__global__ void recording(float *d_vz, float *d_vx, float *d_vy, int nz, int nx, float *d_data_x, float *d_data_y, float *d_data_z,
                          int iShot, int it, int nSteps, int nrec, int *d_z_rec, int *d_x_rec, int *d_y_rec) {
  int iRec = threadIdx.x + blockDim.x * blockIdx.x;
  if (iRec >= nrec) {
    return;
  }
  // if (d_y_rec[iRec] != 23){
  //   printf("d_z_rec[iRec] = %d\n", d_z_rec[iRec]);
  // printf("d_x_rec[iRec] = %d\n", d_x_rec[iRec]);
  // printf("d_y_rec[iRec] = %d\n", d_y_rec[iRec]);
  // }
  d_data_x[(iRec) * (nSteps) + (it)] = d_vx[d_z_rec[iRec] + d_x_rec[iRec] * nz + d_y_rec[iRec] * (nx * nz)];
  d_data_y[(iRec) * (nSteps) + (it)] = d_vy[d_z_rec[iRec] + d_x_rec[iRec] * nz + d_y_rec[iRec] * (nx * nz)];
  d_data_z[(iRec) * (nSteps) + (it)] = d_vz[d_z_rec[iRec] + d_x_rec[iRec] * nz + d_y_rec[iRec] * (nx * nz)];
}

__global__ void res_injection(float *d_vx_adj, float *d_vy_adj, float *d_vz_adj, int nz, int nx, 
                              float *d_res_x, float *d_res_y, float *d_res_z, int it, float dt, int nSteps,
                              int nrec, int *d_z_rec, int *d_x_rec, int *d_y_rec) {
  int iRec = threadIdx.x + blockDim.x * blockIdx.x;
  if (iRec >= nrec) {
    return;
  }
  d_vx_adj[d_z_rec[iRec] + d_x_rec[iRec] * nz + d_y_rec[iRec] * (nx * nz)] += d_res_x[(iRec) * (nSteps) + (it)];
  d_vy_adj[d_z_rec[iRec] + d_x_rec[iRec] * nz + d_y_rec[iRec] * (nx * nz)] += d_res_y[(iRec) * (nSteps) + (it)];
  d_vz_adj[d_z_rec[iRec] + d_x_rec[iRec] * nz + d_y_rec[iRec] * (nx * nz)] += d_res_z[(iRec) * (nSteps) + (it)];
}

// __global__ void source_grad(float *d_szz_adj, float *d_sxx_adj, float *d_syy_adj, int nz, int nx,
//                             float *d_StfGrad, int it, float dt, int z_src,
//                             int x_src, int y_src, double rxz, double rxy) {
//   int id = threadIdx.x + blockDim.x * blockIdx.x;
//   if (id == 0) {
//     d_StfGrad[it] =
//          -(d_szz_adj[z_src+nz*x_src+(nz*nx)*y_src]
//          + d_sxx_adj[z_src+nz*x_src+(nz*nx)*y_src]
//          + d_syy_adj[z_src+nz*x_src+(nz*nx)*y_src]) * dt;
//   } else {
//     return;
//   }
// }

// windowing in the time axis
__global__ void cuda_window(int nt, int nrec, float dt, float *d_win_start,
                            float *d_win_end, float *d_weights,
                            float src_weight, float ratio, float *data) {
  int idt = blockIdx.x * blockDim.x + threadIdx.x;
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int ip = idr * nt + idt;

  // stupid bug... (I put the if just befor line 614)
  if (idt >= 0 && idt < nt && idr >= 0 && idr < nrec) {
    float window_amp = 1.0;

    float t = idt * dt;

    if (ratio > 0.5) {
      printf("Dividing by zero!\n");
      return;
    }

    float t0 = d_win_start[idr];
    float t3 = d_win_end[idr];
    if (t0 == 0.0 && t3 == 0.0) printf("t0 = %f, t3 = %f\n\n", t0, t3);

    float t_max = nt * dt;
    if (t0 < 0.0) t0 = 0.0;
    if (t0 > t_max) t0 = t_max;
    if (t3 < 0.0) t3 = 0.0;
    if (t3 > t_max) t3 = t_max;

    float offset = (t3 - t0) * ratio;
    if (offset <= 0.0) {
      printf("Window error 1!!\n");
      printf("offset = %f\n", offset);
      return;
    }

    float t1 = t0 + offset;
    float t2 = t3 - offset;

    if (t >= t0 && t < t1) {
      window_amp = sin(PI / 2.0 * (t - t0) / (t1 - t0));
    } else if (t >= t1 && t < t2) {
      window_amp = 1.0;
    } else if (t >= t2 && t < t3) {
      window_amp = cos(PI / 2.0 * (t - t2) / (t3 - t2));
    } else {
      window_amp = 0.0;
    }

    data[ip] *= window_amp * window_amp * d_weights[idr] * src_weight;
  } else {
    return;
  }
}
// overloaded window function: without specifying windows and weights
__global__ void cuda_window(int nt, int nrec, float dt, float ratio,
                            float *data) {
  int idt = blockIdx.x * blockDim.x + threadIdx.x;
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int ip = idr * nt + idt;

  if (idt >= 0 && idt < nt && idr >= 0 && idr < nrec) {
    float window_amp = 1.0;

    float t = idt * dt;


    float t0 = 0;
    float t3 = nt * dt;

    float offset = nt * dt * ratio;
    if (2.0 * offset >= t3 - t0) {
      printf("Window error 2!\n");
      return;
    }

    float t1 = t0 + offset;
    float t2 = t3 - offset;

    if (t >= t0 && t < t1) {
      window_amp = sin(PI / 2.0 * (t - t0) / (t1 - t0));
    } else if (t >= t1 && t < t2) {
      window_amp = 1.0;
    } else if (t >= t2 && t < t3) {
      window_amp = cos(PI / 2.0 * (t - t2) / (t3 - t2));
    } else {
      window_amp = 0.0;
    }

    data[ip] *= window_amp * window_amp;
  }
}
