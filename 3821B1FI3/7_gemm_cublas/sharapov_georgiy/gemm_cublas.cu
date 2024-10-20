#include <cmath>
#include <cstddef>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include "gemm_cublas.h"

#define CUDA_CALL(status)                                     \
  {                                                           \
    if (status != cudaSuccess) {                              \
      std::cout << "error " << __FILE__ << ":" << __LINE__;   \
      std::cout << " " << cudaGetErrorString(status) << "\n"; \
      std::exit(0);                                           \
    }                                                         \
  }

#define CUBLAS_CALL(status)                                         \
  {                                                                 \
    if (status != CUBLAS_STATUS_SUCCESS) {                          \
      std::cout << "error " << __FILE__ << ":" << __LINE__ << "\n"; \
      std::exit(0);                                                 \
    }                                                               \
  }

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b, int n) {
  size_t size = n * n;
  size_t byte_count = size * sizeof(float);

  float* dev_a_mem = nullptr;
  float* dev_b_mem = nullptr;

  std::vector<float> result(size);
  float* dev_result_mem = nullptr;
  cublasHandle_t handle;

  float alpha = 1.0f;
  float beta = 0.0f;

  CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_a_mem), byte_count));
  CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_b_mem), byte_count));
  CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_result_mem), byte_count));

  CUDA_CALL(
      cudaMemcpy(dev_a_mem, a.data(), byte_count, cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(dev_b_mem, b.data(), byte_count, cudaMemcpyHostToDevice));

  CUBLAS_CALL(cublasCreate(&handle));
  CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dev_a_mem, n, dev_b_mem, n, &beta, dev_result_mem, n));
  cublasDestroy(handle);

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(result.data()),
                       reinterpret_cast<void*>(dev_result_mem), byte_count,
                       cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(reinterpret_cast<void*>(dev_a_mem)));
  CUDA_CALL(cudaFree(reinterpret_cast<void*>(dev_b_mem)));
  CUDA_CALL(cudaFree(reinterpret_cast<void*>(dev_result_mem)));

  return result;
}
