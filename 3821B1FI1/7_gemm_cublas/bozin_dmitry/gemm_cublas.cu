// Copyright (c) 2024 Bozin Dmitry
#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
  const unsigned _n = n;
  const unsigned sz = _n * _n;
  const unsigned sz_bytes = sz * sizeof(float);
  if (a.size() != sz || b.size() != sz) {
    return std::vector<float>();
  }
  std::vector<float> res(sz);
  float* a_dev;
  float* b_dev;
  float* res_dev;
  cudaMalloc((void**)&a_dev, sz_bytes);
  cudaMalloc((void**)&b_dev, sz_bytes);
  cudaMalloc((void**)&res_dev, sz_bytes);
  cudaMemcpy(a_dev, a.data(), sz_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b.data(), sz_bytes, cudaMemcpyHostToDevice);
  cublasHandle_t handle;
  cublasCreate(&handle);
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
    b_dev, CUDA_R_32F, n,
    a_dev, CUDA_R_32F, n,
    &beta, res_dev, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
  cudaMemcpy(res.data(), res_dev, sz_bytes, cudaMemcpyDeviceToHost);
  return res;
}