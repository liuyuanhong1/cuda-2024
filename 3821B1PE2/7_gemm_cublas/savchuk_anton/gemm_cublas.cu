#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>

void transpose_host(const float* src, float* dst, int n) {
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col)
            dst[col * n + row] = src[row * n + col];
}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate device memory
    float* A_d, * B_d, * C_d;
    cudaMalloc((void**)&A_d, n * n * sizeof(float));
    cudaMalloc((void**)&B_d, n * n * sizeof(float));
    cudaMalloc((void**)&C_d, n * n * sizeof(float));

    // Transpose matrices A and B on the host to convert to column-major order
    std::vector<float> A_t(n * n);
    std::vector<float> B_t(n * n);
    transpose_host(a.data(), A_t.data(), n);
    transpose_host(b.data(), B_t.data(), n);

    // Copy transposed matrices to device memory
    cudaMemcpy(A_d, A_t.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_t.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Perform matrix multiplication using cuBLAS
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N, // No transpose
        CUBLAS_OP_N, // No transpose
        n,           // Number of rows of matrix op(A) and C
        n,           // Number of columns of matrix op(B) and C
        n,           // Number of columns of op(A) and rows of op(B)
        &alpha,
        A_d,         // Device pointer to matrix A
        n,           // Leading dimension of A
        B_d,         // Device pointer to matrix B
        n,           // Leading dimension of B
        &beta,
        C_d,         // Device pointer to matrix C
        n            // Leading dimension of C
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemm failed\n");
    }

    // Copy result from device to host
    std::vector<float> C_t(n * n);
    cudaMemcpy(C_t.data(), C_d, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Transpose the result back to row-major order
    std::vector<float> c(n * n);
    transpose_host(C_t.data(), c.data(), n);

    // Clean up resources
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cublasDestroy(handle);

    return c;
}