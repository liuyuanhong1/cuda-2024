// Copyright 2024 Kachalov Mikhail
#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float> &a,
                              const std::vector<float> &b,
                              int n)
{
    size_t size = n * n * sizeof(float);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    cublasHandle_t handle;
    cublasCreate(&handle);
    float *d_b_t;
    cudaMalloc(&d_b_t, size);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha, d_b, n, &beta, d_b, n, d_b_t, n);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &alpha, d_a, n, d_b_t, n, &beta, d_c, n);

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_b_t);
    cudaFree(d_c);

    cublasDestroy(handle);

    return c;
}