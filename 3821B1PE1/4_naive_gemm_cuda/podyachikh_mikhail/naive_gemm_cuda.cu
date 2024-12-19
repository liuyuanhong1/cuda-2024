// Copyright (c) 2024 Podyachikh Mikhail
#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

__global__ void naiveGemmKernel(const float *a, const float *b,
                                float *c, int n) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    float val = 0.0f;
    for (int k = 0; k < n; ++k) {
      val += a[i * n + k] * b[k * n + j];
    }
    c[i * n + j] = val;
  }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 int n) {
  constexpr int threadsPerDim = 32;
  const int blocksCnt = (n + threadsPerDim - 1) / threadsPerDim;
  const dim3 blocksNum(blocksCnt, blocksCnt);
  const dim3 treadsNum(threadsPerDim, threadsPerDim);

  std::vector<float> c(n * n);
  float *buf_a, *buf_b, *buf_c;
  cudaMalloc(&buf_a, sizeof(float) * n * n);
  cudaMalloc(&buf_b, sizeof(float) * n * n);
  cudaMalloc(&buf_c, sizeof(float) * n * n);
  cudaMemcpy(buf_a, a.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(buf_b, b.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice);

  naiveGemmKernel<<<blocksNum, treadsNum>>>(buf_a, buf_b, buf_c, n);

  cudaMemcpy(c.data(), buf_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

  cudaFree(buf_a);
  cudaFree(buf_b);
  cudaFree(buf_c);

  return c;
}
