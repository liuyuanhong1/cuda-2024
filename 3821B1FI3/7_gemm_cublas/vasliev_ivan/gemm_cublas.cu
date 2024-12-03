#include <cstdlib>
#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "gemm_cublas.h"


#include <chrono>

#define CUDA_CALL(callable) \
  { \
    auto error = callable; \
    if (error != cudaSuccess) { \
      std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
      std::exit(error); \
    } \
  }


#define CUBLAS_CALL(callable) \
  { \
    auto status = callable; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
      std::cerr << "cuBLAS error: " << status << std::endl; \
      std::exit(status); \
    } \
  }

std::vector<float> GemmUsingCUBLAS(const std::vector<float>& matrixA,
                                   const std::vector<float>& matrixB,
                                   int size) {
  cudaDeviceProp deviceProperties{};
  CUDA_CALL(cudaGetDeviceProperties(&deviceProperties, 0));

  size_t matrixSize = size * size;
  if (matrixA.size() != matrixSize || matrixB.size() != matrixSize) return {};

  std::vector<float> resultHost(matrixSize);
  size_t memorySize = matrixSize * sizeof(float);
  float alpha = 1.0f;
  float beta = 0.0f;

  float* deviceA = nullptr;
  float* deviceB = nullptr;
  float* deviceC = nullptr;

  CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&deviceA), memorySize));
  CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&deviceB), memorySize));
  CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&deviceC), memorySize));

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(deviceA), reinterpret_cast<const void*>(matrixA.data()), memorySize, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(deviceB), reinterpret_cast<const void*>(matrixB.data()), memorySize, cudaMemcpyHostToDevice));

  cublasHandle_t cublasHandle{};
  CUBLAS_CALL(cublasCreate(&cublasHandle));


  CUBLAS_CALL(cublasSetMathMode(cublasHandle, CUBLAS_TF32_TENSOR_OP_MATH));


  CUBLAS_CALL(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, deviceB, size, deviceA, size, &beta, deviceC, size));

  CUBLAS_CALL(cublasDestroy(cublasHandle));

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(resultHost.data()), reinterpret_cast<void*>(deviceC), memorySize, cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(reinterpret_cast<void*>(deviceA)));
  CUDA_CALL(cudaFree(reinterpret_cast<void*>(deviceB)));
  CUDA_CALL(cudaFree(reinterpret_cast<void*>(deviceC)));

  return resultHost;
}