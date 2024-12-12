// Copyright (c) 2024 Durandin Vladimir

#include <cstdlib>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "naive_gemm_cuda.h"

__global__ void MatrixMulKernel(const float *a, const float *b, float *c,
                                int n) {
  int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (rowIdx < n && colIdx < n) {
    float partialSum = 0.0f;
    for (int idx = 0; idx < n; ++idx) {
      partialSum += a[rowIdx * n + idx] * b[idx * n + colIdx];
    }
    c[rowIdx * n + colIdx] = partialSum;
  }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b, int n) {
  std::vector<float> resultMatrix(n * n);

  float *deviceA = nullptr;
  float *deviceB = nullptr;
  float *deviceC = nullptr;

  cudaMalloc(&deviceA, static_cast<size_t>(n) * n * sizeof(float));
  cudaMalloc(&deviceB, static_cast<size_t>(n) * n * sizeof(float));
  cudaMalloc(&deviceC, static_cast<size_t>(n) * n * sizeof(float));

  cudaMemcpy(deviceA, a.data(), static_cast<size_t>(n) * n * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, b.data(), static_cast<size_t>(n) * n * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

  MatrixMulKernel<<<numBlocks, threadsPerBlock>>>(deviceA, deviceB, deviceC, n);

  cudaMemcpy(resultMatrix.data(), deviceC,
             static_cast<size_t>(n) * n * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  return resultMatrix;
}
