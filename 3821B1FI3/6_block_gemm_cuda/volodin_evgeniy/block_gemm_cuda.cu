#include "block_gemm_cuda.h"
#include <cassert>

static constexpr int threadsPerDim = 32;

__global__ void BlockGemmCUDA_kernel(const float* a, const float* b, float* c, const int n) {
  __shared__ float a_block[threadsPerDim][threadsPerDim];
  __shared__ float b_block[threadsPerDim][threadsPerDim];

  const int J = blockIdx.x * blockDim.x;
  const int I = blockIdx.y * blockDim.y;
  const int jj = threadIdx.x;
  const int ii = threadIdx.y;
  const int j = J + jj;
  const int i = I + ii;

  float res = 0.0f;

  for (int k_block = 0; k_block < n / threadsPerDim; k_block++) {
    a_block[ii][jj] = a[(i) * n + (threadsPerDim * k_block + jj)];
    b_block[ii][jj] = b[(threadsPerDim * k_block + ii) * n + (j)];
    __syncthreads();
    for (int k = 0; k < threadsPerDim; k++) {
      res += a_block[ii][k] * b_block[k][jj];
    }
    __syncthreads();
  }

  c[i * n + j] = res;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
  assert(n % threadsPerDim == 0);

  const int blocksCnt = n / threadsPerDim;
  const dim3 blocksDim(blocksCnt, blocksCnt);
  const dim3 threadsDim(threadsPerDim, threadsPerDim);

  std::vector<float> c(n * n);
  float* ptr_a;
  float* ptr_b;
  float* ptr_c;
  cudaMalloc(&ptr_a, sizeof(float) * n * n);
  cudaMalloc(&ptr_b, sizeof(float) * n * n);
  cudaMalloc(&ptr_c, sizeof(float) * n * n);
  cudaMemcpy(ptr_a, a.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(ptr_b, b.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice);

  BlockGemmCUDA_kernel<<<blocksDim, threadsDim>>>(ptr_a, ptr_b, ptr_c, n);

  cudaMemcpy(c.data(), ptr_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

  cudaFree(ptr_a);
  cudaFree(ptr_b);
  cudaFree(ptr_c);

  return c;
}