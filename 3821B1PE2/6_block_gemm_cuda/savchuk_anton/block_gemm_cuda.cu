// block_gemm_cuda.cu
#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define BLOCK_SIZE 16  // You can adjust this value based on your GPU's shared memory size

// CUDA kernel for Block Matrix Multiplication
__global__ void block_gemm_kernel(const float* A, const float* B, float* C, int n) {
    // Compute global row and column indices
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Allocate shared memory for A, B, and C tiles
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float Cvalue = 0.0f;

    // Loop over all tiles
    for (int t = 0; t < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Load elements into shared memory if within bounds
        if (row < n && (t * BLOCK_SIZE + threadIdx.x) < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + t * BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if ((t * BLOCK_SIZE + threadIdx.y) < n && col < n)
            Bs[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Cvalue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result
    if (row < n && col < n) {
        C[row * n + col] = Cvalue;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {

    // Allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;

    size_t size = n * n * sizeof(float);

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), size, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    block_gemm_kernel << <dimGrid, dimBlock >> > (d_A, d_B, d_C, n);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return c;
}