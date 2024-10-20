// Copyright (c) 2024 Ivanov Nikita
#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    std::vector<float> c(n * n, 0.0f);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float* d_a;
    float* d_b;
    float* d_c;
    size_t sizeInBytes = n * n * sizeof(float);
    cudaMalloc(&d_a, sizeInBytes);
    cudaMalloc(&d_b, sizeInBytes);
    cudaMalloc(&d_c, sizeInBytes);

    cudaMemcpy(d_a, a.data(), sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeInBytes, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);

    cudaMemcpy(c.data(), d_c, sizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cublasDestroy(handle);

    return c;
}
