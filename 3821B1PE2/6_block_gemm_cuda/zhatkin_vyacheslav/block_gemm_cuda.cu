#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    // Ensure input matrices are valid
    if (a.size() != n * n || b.size() != n * n) {
        throw std::invalid_argument("Matrix size mismatch");
    }

    // Allocate memory on GPU for input matrices and result matrix
    float* d_A;
    float* d_B;
    float* d_C;
    size_t matrix_size = n * n * sizeof(float);

    cudaMalloc((void**)&d_A, matrix_size);
    cudaMalloc((void**)&d_B, matrix_size);
    cudaMalloc((void**)&d_C, matrix_size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, a.data(), matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), matrix_size, cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set up scaling factors
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    // Note: cuBLAS assumes column-major order, so we transpose the inputs
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose for both matrices
        n, n, n,                   // Dimensions of the matrices
        &alpha,                    // Scaling factor for A * B
        d_A, n,                    // Matrix A and leading dimension
        d_B, n,                    // Matrix B and leading dimension
        &beta,                     // Scaling factor for C
        d_C, n);                   // Matrix C and leading dimension

    // Allocate result vector on host and copy result from device to host
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_C, matrix_size, cudaMemcpyDeviceToHost);

    // Free GPU memory and destroy cuBLAS handle
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return c;
}