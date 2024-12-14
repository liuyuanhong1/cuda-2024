// Copyright (c) 2024 Vinichuk Timofey

#include "block_gemm_cuda.h"
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

__global__ void BlockGemmKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float shared_a[SIZE][SIZE];
    __shared__ float shared_b[SIZE][SIZE];

    int row = blockIdx.y * SIZE + threadIdx.y;
    int col = blockIdx.x * SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < n; k += SIZE) {
        if (row < n && k + threadIdx.y < n) {
            shared_a[threadIdx.y][threadIdx.x] = a[row * n + k + threadIdx.x];
        }
        else {
            shared_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < n && k + threadIdx.x < n) {
            shared_b[threadIdx.y][threadIdx.x] = b[(k + threadIdx.y) * n + col];
        }
        else {
            shared_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < SIZE; ++i) {
            sum += shared_a[threadIdx.y][i] * shared_b[i][threadIdx.x];
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

    float* kernel_a = nullptr;
    float* kernel_b = nullptr;
    float* kernel_c = nullptr;

    cudaMalloc(&kernel_a, n * n * sizeof(float));
    cudaMalloc(&kernel_b, n * n * sizeof(float));
    cudaMalloc(&kernel_c, n * n * sizeof(float));
    cudaMemcpy(kernel_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(SIZE, SIZE);
    dim3 numBlocks((n + SIZE - 1) / SIZE, (n + SIZE - 1) / SIZE);


    BlockGemmKernel << < numBlocks, blockSize >> > (kernel_a, kernel_b, kernel_c, n);
    cudaMemcpy(c.data(), kernel_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(kernel_a);
    cudaFree(kernel_b);
    cudaFree(kernel_c);

    return c;
}