// Copyright (c) 2024 Kirillov Maxim
#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

#define SIZE 32

__global__ void NaiveGemmKernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}


std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(n * n, 0.0f);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    size_t sizeInBytes = n * n * sizeof(float);
    cudaMalloc((void**)&d_a, sizeInBytes);
    cudaMalloc((void**)&d_b, sizeInBytes);
    cudaMalloc((void**)&d_c, sizeInBytes);

    cudaMemcpy(d_a, a.data(), sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeInBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(SIZE, SIZE);
    dim3 blocksCount((n + SIZE - 1) / SIZE, (n + SIZE - 1) / SIZE);

    NaiveGemmKernel<<<blocksCount, blockSize>>>(d_a, d_b, d_c, n);
    cudaMemcpy(c.data(), d_c, sizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}