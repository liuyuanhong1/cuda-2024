// Copyright (c) 2024 Khodyrev Fedor
#include "block_gemm_cuda.h"
#include <cuda_runtime.h>


__global__ void BlockGemmKernel(const float* var1, const float* var2, float* var3, int n, int block_size) {
    __shared__ float shared_a[16][16];
    __shared__ float shared_b[16][16];

    int row = blockIdx.y * block_size + threadIdx.y;
    int col = blockIdx.x * block_size + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < n; k += block_size) {
        if (row < n && k + threadIdx.y < n) {
            shared_a[threadIdx.y][threadIdx.x] = var1[row * n + k + threadIdx.x];
        } else {
            shared_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < n && k + threadIdx.x < n) {
            shared_b[threadIdx.y][threadIdx.x] = var2[(k + threadIdx.y) * n + col];
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
        var3[row * n + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& var1, const std::vector<float>& var2, int n) {
    auto countElem = n * n;
    if (var1.size() != countElem || var2.size() != countElem) return {};

    std::vector<float> var3(countElem, 0.0f);

    float* d_a;
    float* d_b;
    float* d_c;
    cudaMalloc(&d_a, countElem * sizeof(float));
    cudaMalloc(&d_b, countElem * sizeof(float));
    cudaMalloc(&d_c, countElem * sizeof(float));

    cudaMemcpy(d_a, var1.data(), countElem * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, var2.data(), countElem * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 16;
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);

    BlockGemmKernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n, block_size);
    cudaMemcpy(var3.data(), d_c, countElem * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return var3;
}