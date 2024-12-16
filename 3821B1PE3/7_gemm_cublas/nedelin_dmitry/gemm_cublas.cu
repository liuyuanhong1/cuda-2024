// Copyright (c) 2024 Nedelin Dmitry

#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "gemm_cublas.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& matrixA,
                              const std::vector<float>& matrixB,
                              int dimension)
{
    const float scalarAlpha = 1.0f;
    const float scalarBeta = 0.0f;

    const auto totalElements = dimension * dimension;
    std::vector<float> resultMatrix(totalElements);
    const auto totalBytes = totalElements * sizeof(float);

    float* deviceMatrixA = nullptr;
    cudaMalloc(&deviceMatrixA, totalBytes);
    float* deviceMatrixB = nullptr;
    cudaMalloc(&deviceMatrixB, totalBytes);
    float* deviceMatrixC = nullptr;
    cudaMalloc(&deviceMatrixC, totalBytes);

    cudaMemcpy(deviceMatrixA, matrixA.data(), totalBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, matrixB.data(), totalBytes, cudaMemcpyHostToDevice);

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    cublasSetMathMode(cublasHandle, CUBLAS_TF32_TENSOR_OP_MATH);

    cublasGemmEx(cublasHandle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 dimension, dimension, dimension,
                 &scalarAlpha,
                 deviceMatrixB, CUDA_R_32F, dimension,
                 deviceMatrixA, CUDA_R_32F, dimension,
                 &scalarBeta,
                 deviceMatrixC, CUDA_R_32F, dimension,
                 CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);

    cudaMemcpy(resultMatrix.data(), deviceMatrixC, totalBytes, cudaMemcpyDeviceToHost);

    cublasDestroy(cublasHandle);
    cudaFree(deviceMatrixC);
    cudaFree(deviceMatrixB);
    cudaFree(deviceMatrixA);

    return resultMatrix;
}
