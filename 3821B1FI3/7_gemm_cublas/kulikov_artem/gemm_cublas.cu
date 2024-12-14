// Copyright (c) 2024 Kulikov Artem
#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdlib>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b, int size) {
    std::vector<float> c(size * size);

    size_t sizeInBytes = size * size * sizeof(*a.data());

    float* d_a;
    cudaMalloc(&d_a, sizeInBytes);
    float* d_b;
    cudaMalloc(&d_b, sizeInBytes);
    float* d_c;
    cudaMalloc(&d_c, sizeInBytes);

    cudaMemcpy(d_a, a.data(), sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeInBytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 size, size, size,
                 &alpha,
                 d_b, CUDA_R_32F, size,
                 d_a, CUDA_R_32F, size,
                 &beta,
                 d_c, CUDA_R_32F, size,
                 CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);

    cudaMemcpy(c.data(), d_c, sizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return c;
}
