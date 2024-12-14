// Copyright (c) 2024 Durandin Vladimir

#include <cstdlib>
#include <iostream>

#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gemm_cublas.h"

#define VERIFY_CUDA_CALL(call)                                                 \
  {                                                                            \
    auto errorCode = call;                                                     \
    if (errorCode != cudaSuccess) {                                            \
      std::cerr << "\033[1;31mCUDA Error:\033[0m ";                            \
      std::cerr << cudaGetErrorString(errorCode) << '\n';                      \
      std::cerr << "Error code: " << static_cast<int>(errorCode) << '\n';      \
      std::cerr << "Location: " << __FILE__ << " (" << __LINE__ << ")\n";      \
      std::exit(errorCode);                                                    \
    }                                                                          \
  }

#define VERIFY_CUBLAS_CALL(call)                                               \
  {                                                                            \
    auto cublasStatus = call;                                                  \
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {                               \
      std::cerr << "\033[1;31mcuBLAS Error:\033[0m ";                          \
      std::cerr << static_cast<int>(cublasStatus) << '\n';                     \
      std::cerr << "Location: " << __FILE__ << " (" << __LINE__ << ")\n";      \
      std::exit(cublasStatus);                                                 \
    }                                                                          \
  }

std::vector<float> GemmCUBLAS(const std::vector<float> &a,
                              const std::vector<float> &b, int size) {
  cudaDeviceProp deviceProperties{};
  VERIFY_CUDA_CALL(cudaGetDeviceProperties(&deviceProperties, 0));

  size_t totalElements = size * size;
  if (a.size() != totalElements || b.size() != totalElements)
    return {};

  std::vector<float> resultHost(totalElements);
  auto totalBytes = totalElements * sizeof(float);
  float alphaVal = 1.0f;
  float betaVal = 0.0f;

  float *devA = nullptr;
  float *devB = nullptr;
  float *devC = nullptr;

  VERIFY_CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&devA), totalBytes));
  VERIFY_CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&devB), totalBytes));
  VERIFY_CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&devC), totalBytes));

  VERIFY_CUDA_CALL(cudaMemcpy(reinterpret_cast<void *>(devA),
                              reinterpret_cast<const void *>(a.data()),
                              totalBytes, cudaMemcpyHostToDevice));
  VERIFY_CUDA_CALL(cudaMemcpy(reinterpret_cast<void *>(devB),
                              reinterpret_cast<const void *>(b.data()),
                              totalBytes, cudaMemcpyHostToDevice));

  cublasHandle_t cublasHandle{};
  VERIFY_CUBLAS_CALL(cublasCreate(&cublasHandle));
  VERIFY_CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, size,
                                 size, size, &alphaVal, devB, size, devA, size,
                                 &betaVal, devC, size));
  VERIFY_CUBLAS_CALL(cublasDestroy(cublasHandle));

  VERIFY_CUDA_CALL(cudaMemcpy(reinterpret_cast<void *>(resultHost.data()),
                              reinterpret_cast<void *>(devC), totalBytes,
                              cudaMemcpyDeviceToHost));

  VERIFY_CUDA_CALL(cudaFree(reinterpret_cast<void *>(devA)));
  VERIFY_CUDA_CALL(cudaFree(reinterpret_cast<void *>(devB)));
  VERIFY_CUDA_CALL(cudaFree(reinterpret_cast<void *>(devC)));

  return resultHost;
}
