#include <cstdlib>
#include <iostream>
#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gemm_cublas.h"

#define CUDA_CHECK(error) \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CUBLAS_CHECK(status) \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << status << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }


std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
  cudd_aiceProp deviceProp{};
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

  size_t countElem = n * n;
  if (a.size() != countElem || b.size() != countElem) return {};

  std::vector<float> c(countElem);
  auto bytes = countElem * sizeof(float);
  float alpha = 1.0f;
  float beta = 0.0f;

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), bytes));

  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_a), reinterpret_cast<const void*>(a.data()), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_b), reinterpret_cast<const void*>(b.data()), bytes, cudaMemcpyHostToDevice));

  cublasHandle_t handle{};
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_b, n, d_a, n, &beta, d_c, n));
  CUBLAS_CHECK(cublasDestroy(handle));

  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(c.data()), reinterpret_cast<void*>(d_c), bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_a)));
  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_b)));
  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_c)));

  return c;
}