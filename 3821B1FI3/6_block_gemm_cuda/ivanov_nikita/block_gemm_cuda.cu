// Copyright (c) 2024 Ivanov Nikita
#include "block_gemm_cuda.h"
#include <cuda_runtime.h>


__global__ void BlockGemmKernel(const float* a, const float* b, float* c, int n, int block_size) {
    __shared__ float shared_a[16][16];
    __shared__ float shared_b[16][16];

    int row = blockIdx.y * block_size + threadIdx.y;
    int col = blockIdx.x * block_size + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < n; k += block_size) {
        if (row < n && k + threadIdx.y < n) {
            shared_a[threadIdx.y][threadIdx.x] = a[row * n + k + threadIdx.x];
        } else {
            shared_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < n && k + threadIdx.x < n) {
            shared_b[threadIdx.y][threadIdx.x] = b[(k + threadIdx.y) * n + col];
        } else {
            shared_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < block_size; ++i) {
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

    float* d_a;
    float* d_b;
    float* d_c;
    cudaMalloc(&d_a, n * n * sizeof(float));
    cudaMalloc(&d_b, n * n * sizeof(float));
    cudaMalloc(&d_c, n * n * sizeof(float));

    cudaMemcpy(d_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 16;
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);

    BlockGemmKernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n, block_size);
    cudaMemcpy(c.data(), d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
