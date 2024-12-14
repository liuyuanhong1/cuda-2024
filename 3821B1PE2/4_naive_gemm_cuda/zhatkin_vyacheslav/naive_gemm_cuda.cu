#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// CUDA Kernel for matrix multiplication
__global__ void MatrixMultiplyKernel(const float* a, const float* b, float* c, int n) {
    // Calculate the row and column of the current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    // Total size of matrices
    size_t size = n * n * sizeof(float);

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy matrices to device
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    MatrixMultiplyKernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // Allocate host memory for the result
    std::vector<float> c(n * n);

    // Copy result back to host
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
