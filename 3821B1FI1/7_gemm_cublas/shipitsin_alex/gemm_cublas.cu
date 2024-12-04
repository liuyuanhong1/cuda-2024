// Copyright (c) 2024 Shipitsin Alex 

#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "gemm_cublas.h"


std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b, 
                              int size) 
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const auto countElem = size * size;
    std::vector<float> output(countElem);
    const auto sizeInBytes = countElem * sizeof(float);

    float *aDev = nullptr;
    cudaMalloc(&aDev, sizeInBytes);
    float *bDev = nullptr;
    cudaMalloc(&bDev, sizeInBytes);
    float *cDev = nullptr;
    cudaMalloc(&cDev, sizeInBytes);

    cudaMemcpy(aDev, a.data(), sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bDev, b.data(), sizeInBytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 size, size, size,
                 &alpha,
                 bDev, CUDA_R_32F, size,
                 aDev, CUDA_R_32F, size,
                 &beta,
                 cDev, CUDA_R_32F, size,
                 CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);

    cudaMemcpy(output.data(), cDev, sizeInBytes, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(cDev);
    cudaFree(bDev);
    cudaFree(aDev);

    return output;
}
