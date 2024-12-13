// Copyright (c) 2024 Tushentsova Karina
#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    int totalElements = n * n;
    size_t sizeBytes = totalElements * sizeof(float);
    std::vector<float> output(totalElements, 0.0f);

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceOutput = nullptr;

    cudaMalloc(&deviceA, sizeBytes);
    cudaMalloc(&deviceB, sizeBytes);
    cudaMalloc(&deviceOutput, sizeBytes);

    cudaMemcpy(deviceA, a.data(), sizeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b.data(), sizeBytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, deviceB, n, deviceA, n, &beta, deviceOutput, n);
    cudaMemcpy(output.data(), deviceOutput, sizeBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceOutput);
    cublasDestroy(handle);

    return output;
}