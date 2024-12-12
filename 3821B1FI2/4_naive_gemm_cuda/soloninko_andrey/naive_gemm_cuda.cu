// Copyright (c) 2024 Soloninko Andrey
#include <cstdlib>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "naive_gemm_cuda.h"

__global__ void MatrixMulKernel(const float* a, const float* b, float* c,
                                int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
      sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
  }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, int n) {
  
  const size_t count = n * n;
  std::vector<float> c(n * n);

  float* d_a;
  float* d_b;
  float* d_c;

  cudaMalloc(&d_a, count * sizeof(float));
  cudaMalloc(&d_b, count * sizeof(float));
  cudaMalloc(&d_c, count * sizeof(float));

  cudaMemcpy(d_a, a.data(),
             count * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(),
             count * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

  MatrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);

  cudaMemcpy(c.data(), d_c,
             count * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return c;
}
