//  Copyright (c) 2024 Vinokurov Ivan
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "gemm_cublas.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n)
{
    const float scalarAlpha = 1.0f;
    const float scalarBeta = 0.0f;

    const auto totalElements = n * n;
    std::vector<float> output(totalElements);
    const auto sizeInBytes = totalElements * sizeof(float);

    float* deviceMtxA = nullptr;
    cudaMalloc(&deviceMtxA, sizeInBytes);
    float* deviceMtxB = nullptr;
    cudaMalloc(&deviceMtxB, sizeInBytes);
    float* deviceMtxC = nullptr;
    cudaMalloc(&deviceMtxC, sizeInBytes);

    cudaMemcpy(deviceMtxA, a.data(), sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMtxB, b.data(), sizeInBytes, cudaMemcpyHostToDevice);

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    cublasSetMathMode(cublasHandle, CUBLAS_TF32_TENSOR_OP_MATH);

    cublasGemmEx(cublasHandle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 n, n, n,
                 &scalarAlpha,
                 deviceMtxB, CUDA_R_32F, n,
                 deviceMtxA, CUDA_R_32F, n,
                 &scalarBeta,
                 deviceMtxC, CUDA_R_32F, n,
                 CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);

    cudaMemcpy(output.data(), deviceMtxC, sizeInBytes, cudaMemcpyDeviceToHost);

    cublasDestroy(cublasHandle);
    cudaFree(deviceMtxC);
    cudaFree(deviceMtxB);
    cudaFree(deviceMtxA);

    return output;
}
