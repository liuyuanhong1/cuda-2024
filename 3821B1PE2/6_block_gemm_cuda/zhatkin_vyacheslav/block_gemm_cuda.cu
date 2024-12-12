#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define TILE_SIZE 16 // Block size

__global__ void BlockGemmKernel(const float* A, const float* B, float* C, int n) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0;

    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        int A_row = row;
        int A_col = t * TILE_SIZE + threadIdx.x;
        if (A_row < n && A_col < n)
            sharedA[threadIdx.y][threadIdx.x] = A[A_row * n + A_col];
        else
            sharedA[threadIdx.y][threadIdx.x] = 0;

        int B_row = t * TILE_SIZE + threadIdx.y;
        int B_col = col;
        if (B_row < n && B_col < n)
            sharedB[threadIdx.y][threadIdx.x] = B[B_row * n + B_col];
        else
            sharedB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        // Compute partial results
        for (int k = 0; k < TILE_SIZE; ++k) {
            value += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to global memory
    if (row < n && col < n) {
        C[row * n + col] = value;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    // Allocate device memory
    float* d_A, * d_B, * d_C;
    size_t size = n * n * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    BlockGemmKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // Wait for GPU to finish before accessing on host
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