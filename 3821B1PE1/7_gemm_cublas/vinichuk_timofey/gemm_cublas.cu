// Copyright (c) 2024 Vinichuk Timofey
#include "gemm_cublas.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {

    std::vector<float> c(n * n, 0.0f);
    size_t bytesSize = n * n * sizeof(float);

    cublasHandle_t handle;

    cublasCreate(&handle);

    float* data_a;
    float* data_b;
    float* data_c;

    cudaMalloc(&data_a, bytesSize);
    cudaMalloc(&data_b, bytesSize);
    cudaMalloc(&data_c, bytesSize);

    cudaMemcpy(data_a, a.data(), bytesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(data_b, b.data(), bytesSize, cudaMemcpyHostToDevice);

    const float ALPHA = 1.0f;
    const float BETA = 0.0f;

    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &ALPHA,
        data_b, CUDA_R_32F, n,
        data_a, CUDA_R_32F, n,
        &BETA,
        data_c, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT
    );

    cudaMemcpy(c.data(), data_c, bytesSize, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);

    cudaFree(data_a);
    cudaFree(data_b);
    cudaFree(data_c);

    return c;
}