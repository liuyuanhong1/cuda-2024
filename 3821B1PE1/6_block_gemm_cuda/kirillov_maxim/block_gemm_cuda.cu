// Copyright (c) 2024 Kirillov Maxim
#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#define SIZE 16

__global__ void BlockGemmKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float sharedA[SIZE][SIZE];
    __shared__ float sharedB[SIZE][SIZE];

    int row = blockIdx.y * SIZE + threadIdx.y;
    int col = blockIdx.x * SIZE + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < n / SIZE; ++k) {
        sharedA[threadIdx.y][threadIdx.x] = a[row * n + (k * SIZE + threadIdx.x)];
        sharedB[threadIdx.y][threadIdx.x] = b[(k * SIZE + threadIdx.y) * n + col];

        __syncthreads();

        for (int i = 0; i < SIZE; ++i) {
            sum += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(n * n, 0.0f);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    size_t sizeInBytes = n * n * sizeof(float);

    cudaMalloc(&d_a, sizeInBytes);
    cudaMalloc(&d_b, sizeInBytes);
    cudaMalloc(&d_c, sizeInBytes);

    cudaMemcpy(d_a, a.data(), sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeInBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(SIZE, SIZE);
    dim3 blocksCount((n + SIZE - 1) / SIZE, (n + SIZE - 1) / SIZE);

    BlockGemmKernel<<<blocksCount, blockSize >>>(d_a, d_b, d_c, n);
    cudaMemcpy(c.data(), d_c, sizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}