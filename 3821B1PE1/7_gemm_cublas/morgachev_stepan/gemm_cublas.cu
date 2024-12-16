// Copyright (c) 2024 Morgachev Stepan
#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    size_t sizeInBytes = n * n * sizeof(float);
    std::vector<float> output(n * n, 0.0f);

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceOutput = nullptr;

    cudaMalloc(&deviceA, sizeInBytes);
    cudaMalloc(&deviceB, sizeInBytes);
    cudaMalloc(&deviceOutput, sizeInBytes);

    cudaMemcpy(deviceA, a.data(), sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b.data(), sizeInBytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                deviceB, n,
                deviceA, n,
                &beta,
                deviceOutput, n);

    cudaMemcpy(output.data(), deviceOutput, sizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceOutput);
    cublasDestroy(handle);

    return output;
}
