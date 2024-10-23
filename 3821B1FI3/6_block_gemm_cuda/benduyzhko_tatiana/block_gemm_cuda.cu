#include "block_gemm_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <vector>

#define BLOCK_SIZE 32

__global__ void myKernel(const float* a, const float* b, float* const c,
                         const int size) {
  __shared__ float aCached[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float bCached[BLOCK_SIZE][BLOCK_SIZE];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int nIdx = blockIdx.y * BLOCK_SIZE + ty;
  const int mIdx = blockIdx.x * BLOCK_SIZE + tx;

  float cVal = 0.0f;

  for (int t = 0; t < size / BLOCK_SIZE; ++t) {
    aCached[ty][tx] = a[nIdx * size + t * BLOCK_SIZE + tx];
    bCached[ty][tx] = b[(t * BLOCK_SIZE + ty) * size + mIdx];

    __syncthreads();
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      cVal += aCached[ty][k] * bCached[k][tx];
    }
    __syncthreads();
  }

  if (nIdx < size && mIdx < size) {
    c[nIdx * size + mIdx] = cVal;
  }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, int n) {
  std::vector<float> c(n * n);

  size_t sizeInBytes = n * n * sizeof(*a.data());

  float* device_a;
  float* device_b;
  float* device_c;
  cudaMalloc(&device_a, sizeInBytes);
  cudaMalloc(&device_b, sizeInBytes);
  cudaMalloc(&device_c, sizeInBytes);

  cudaMemcpy(device_a, a.data(), sizeInBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, b.data(), sizeInBytes, cudaMemcpyHostToDevice);

  const int sizeAxis = BLOCK_SIZE;
  dim3 threadsPerBlock(sizeAxis, sizeAxis);
  dim3 numBlocks((n + sizeAxis - 1) / sizeAxis,
                 (n + sizeAxis - 1) / sizeAxis);

  myKernel<<<numBlocks, threadsPerBlock>>>(device_a, device_b, device_c, n);

  cudaMemcpy(c.data(), device_c, sizeInBytes, cudaMemcpyDeviceToHost);

  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
  return c;
}
