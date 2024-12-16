#include "gemm_cublas.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
  std::vector<float> c(n * n);
  float* ptr_a;
  float* ptr_b;
  float* ptr_c;
  cudaMalloc(&ptr_a, sizeof(float) * n * n);
  cudaMalloc(&ptr_b, sizeof(float) * n * n);
  cudaMalloc(&ptr_c, sizeof(float) * n * n);
  cudaMemcpy(ptr_a, a.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(ptr_b, b.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, ptr_b, n, ptr_a, n, &beta, ptr_c, n);
  cublasDestroy(handle);

  cudaMemcpy(c.data(), ptr_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

  cudaFree(ptr_a);
  cudaFree(ptr_b);
  cudaFree(ptr_c);

  return c;
}
