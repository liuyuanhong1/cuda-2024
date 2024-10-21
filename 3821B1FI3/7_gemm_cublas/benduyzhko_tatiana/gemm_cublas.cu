#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdlib>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
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

  cublasHandle_t handle;
  cublasCreate(&handle);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, device_b,
               CUDA_R_32F, n, device_a, CUDA_R_32F, n, &beta, device_c, CUDA_R_32F,
               n, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);

  cudaMemcpy(c.data(), device_c, sizeInBytes, cudaMemcpyDeviceToHost);

  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
  return c;
}
