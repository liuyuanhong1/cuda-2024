// Copyright (c) 2024 Sokolova Daria
#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& matrixA,
                              const std::vector<float>& matrixB,
                              int matrixDim) {
    size_t bufferSize = matrixDim * matrixDim * sizeof(float);
    std::vector<float> resultMatrix(matrixDim * matrixDim, 0.0f);

    float* deviceMatrixA = nullptr;
    float* deviceMatrixB = nullptr;
    float* deviceMatrixC = nullptr;

    cudaMalloc(&deviceMatrixA, bufferSize);
    cudaMalloc(&deviceMatrixB, bufferSize);
    cudaMalloc(&deviceMatrixC, bufferSize);

    cudaMemcpy(deviceMatrixA, matrixA.data(), bufferSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, matrixB.data(), bufferSize, cudaMemcpyHostToDevice);

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    const float scaleAlpha = 1.0f;
    const float scaleBeta = 0.0f;

    cublasSgemm(cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                matrixDim, matrixDim, matrixDim,
                &scaleAlpha,
                deviceMatrixB, matrixDim,
                deviceMatrixA, matrixDim,
                &scaleBeta,
                deviceMatrixC, matrixDim);

    cudaMemcpy(resultMatrix.data(), deviceMatrixC, bufferSize, cudaMemcpyDeviceToHost);

    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceMatrixC);
    cublasDestroy(cublasHandle);

    return resultMatrix;
}
