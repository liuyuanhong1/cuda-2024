// Copyright (c) 2024 Kirillov Maxim
#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    std::vector<float> c(n * n, 0.0f);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    size_t sizeInBytes = n * n * sizeof(float);

    cudaMalloc(&d_a, sizeInBytes);
    cudaMalloc(&d_b, sizeInBytes);
    cudaMalloc(&d_c, sizeInBytes);

    cudaMemcpy(d_a, a.data(), sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeInBytes, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 n, n, n,
                 &alpha,
                 d_a, CUDA_R_32F, n,
                 d_b, CUDA_R_32F, n,
                 &beta,
                 d_c, CUDA_C_32F, n,
                 CUBLAS_COMPUTE_32F_FAST_16F,
                 CUBLAS_GEMM_DEFAULT);

    cudaMemcpy(c.data(), d_c, sizeInBytes, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}