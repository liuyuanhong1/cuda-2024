// Copyright (c) 2024 Moiseev Nikita
#include <cstdlib>
#include <iostream>

#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gemm_cublas.h"

#define CHECK_CUDA_ERROR(callable)                                        \
  {                                                                       \
    auto codeError = callable;                                            \
    if (codeError != cudaSuccess) {                                       \
      std::cerr << "\033[1;31merror\033[0m: ";                            \
      std::cerr << cudaGetErrorString(codeError) << '\n';                 \
      std::cerr << "code error: " << static_cast<int>(codeError) << '\n'; \
      std::cerr << "loc: " << __FILE__ << '(' << __LINE__ << ")\n";       \
      std::exit(codeError);                                               \
    }                                                                     \
  }

#define CHECK_CUBLAS_STATUS(callable)                               \
  {                                                                 \
    auto status = callable;                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                          \
      std::cerr << "\033[1;31mcublas status failed:\033[0m: ";      \
      std::cerr << static_cast<int>(status) << '\n';                \
      std::cerr << "loc: " << __FILE__ << '(' << __LINE__ << ")\n"; \
      std::exit(status);                                            \
    }                                                               \
  }

std::vector<float> GemmCUBLAS(const std::vector<float>& matrix_a, const std::vector<float>& matrix_b, int matrix_size) {
  cudaDeviceProp device_properties{};
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&device_properties, 0));

  size_t total_elements = matrix_size * matrix_size;
  if (matrix_a.size() != total_elements || matrix_b.size() != total_elements) return {};

  std::vector<float> matrix_c_host(total_elements);
  auto total_bytes = total_elements * sizeof(float);
  float alpha = 1.0f;
  float beta = 0.0f;

  float* matrix_a_device = nullptr;
  float* matrix_b_device = nullptr;
  float* matrix_c_device = nullptr;

  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&matrix_a_device), total_bytes));
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&matrix_b_device), total_bytes));
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&matrix_c_device), total_bytes));

  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void*>(matrix_a_device), reinterpret_cast<const void*>(matrix_a.data()), total_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void*>(matrix_b_device), reinterpret_cast<const void*>(matrix_b.data()), total_bytes, cudaMemcpyHostToDevice));

  cublasHandle_t cublas_handle{};
  CHECK_CUBLAS_STATUS(cublasCreate(&cublas_handle));
  CHECK_CUBLAS_STATUS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size, matrix_size, matrix_size, &alpha, matrix_b_device, matrix_size, matrix_a_device, matrix_size, &beta, matrix_c_device, matrix_size));
  CHECK_CUBLAS_STATUS(cublasDestroy(cublas_handle));

  CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void*>(matrix_c_host.data()), reinterpret_cast<void*>(matrix_c_device), total_bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(reinterpret_cast<void*>(matrix_a_device)));
  CHECK_CUDA_ERROR(cudaFree(reinterpret_cast<void*>(matrix_b_device)));
  CHECK_CUDA_ERROR(cudaFree(reinterpret_cast<void*>(matrix_c_device)));

  return matrix_c_host;
}
