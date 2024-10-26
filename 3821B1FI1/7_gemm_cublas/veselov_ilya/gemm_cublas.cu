#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    size_t matrixSize = n * n * sizeof(float);

    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_B, matrixSize);
    cudaMalloc((void**)&d_C, matrixSize);

    cudaMemcpy(d_A, a.data(), matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), matrixSize, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_T,
                n, n, n,
                &alpha,
                d_B, n,
                d_A, n,
                &beta,
                d_C, n);

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_C, matrixSize, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return c;
}
