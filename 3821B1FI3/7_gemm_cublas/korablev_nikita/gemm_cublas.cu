// Copyright (c) 2024 Korablev Nikita
#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    std::vector<float> c(n * n, 0.0f);

    size_t size = n * n * sizeof(float);
    float* d_a;
    float* d_b;
    float* d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 n, n, n, &alpha,
                 d_b, CUDA_R_32F, n,
                 d_a, CUDA_R_32F, n,
                 &beta, d_c, CUDA_R_32F, n,
                 CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cublasDestroy(handle);
    return c;
}
