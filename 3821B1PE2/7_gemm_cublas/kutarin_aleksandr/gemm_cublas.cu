//Copyright Kutarin Aleksandr 2024

#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// Function to multiply two matrices using cuBLAS
std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (a.size() != n * n || b.size() != n * n) {
        throw std::invalid_argument("Matrix dimensions do not match the expected size.");
    }

    // Result matrix
    std::vector<float> c(n * n, 0);

    // Pointers for device memory
    float *d_a, *d_b, *d_c;

    // Allocate device memory
    cudaMalloc((void**)&d_a, n * n * sizeof(float));
    cudaMalloc((void**)&d_b, n * n * sizeof(float));
    cudaMalloc((void**)&d_c, n * n * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // cuBLAS constants for matrix multiplication
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    // Note: cuBLAS expects column-major storage
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_b, n,  // Matrix B
                d_a, n,  // Matrix A
                &beta,
                d_c, n); // Matrix C

    // Copy result back to host
    cudaMemcpy(c.data(), d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);

    return c;
}
