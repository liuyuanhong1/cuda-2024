// Copyright (c) 2024 Podyachikh Mikhail
#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

std::vector<float> GemmCUBLAS(const std::vector<float> &a,
                              const std::vector<float> &b,
                              int n) {
  std::vector<float> c(n * n);

  float *buf_a, *buf_b, *buf_c;
  cudaMalloc(&buf_a, n * n * sizeof(float));
  cudaMalloc(&buf_b, n * n * sizeof(float));
  cudaMalloc(&buf_c, n * n * sizeof(float));

  cudaMemcpy(buf_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(buf_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, buf_b,
              n, buf_a, n, &beta, buf_c, n);
  cublasDestroy(handle);

  cudaMemcpy(c.data(), buf_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(buf_a);
  cudaFree(buf_b);
  cudaFree(buf_c);

  return c;
}
