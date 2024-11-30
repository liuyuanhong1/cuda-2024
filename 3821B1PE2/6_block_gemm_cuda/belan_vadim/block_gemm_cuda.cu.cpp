#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <iostream>

#define BLOCK_SIZE 16

__global__ void blockGemmKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float Cvalue = 0.0f;

    int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        if (row < n && (t * BLOCK_SIZE + threadIdx.x) < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + t * BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < n && (t * BLOCK_SIZE + threadIdx.y) < n)
            Bs[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Cvalue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = Cvalue;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (a.size() != static_cast<size_t>(n * n) || b.size() != static_cast<size_t>(n * n)) {
        throw std::invalid_argument("Input matrices must be of size n x n.");
    }

    std::vector<float> c(n * n, 0.0f);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    size_t size = n * n * sizeof(float);

    cudaError_t err;
    err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for matrix A.");
    }

    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        throw std::runtime_error("Failed to allocate device memory for matrix B.");
    }

    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        throw std::runtime_error("Failed to allocate device memory for matrix C.");
    }

    err = cudaMemcpy(d_A, a.data(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("Failed to copy matrix A to device.");
    }

    err = cudaMemcpy(d_B, b.data(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("Failed to copy matrix B to device.");
    }

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    blockGemmKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("Failed to launch blockGemmKernel.");
    }

    err = cudaMemcpy(c.data(), d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        throw std::runtime_error("Failed to copy matrix C from device to host.");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return c;
}
