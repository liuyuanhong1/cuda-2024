// Copyright (c) 2024 Korablev Nikita
#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

__global__ void NaiveGemmKernel(const float* a, const float* b, float* c, int n) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (size_t k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const size_t count = n*n;
    std::vector<float> c(count, 0.0f);

    float* d_a;
    float* d_b;
    float* d_c;
    cudaMalloc(&d_a, count * sizeof(float));
    cudaMalloc(&d_b, count * sizeof(float));
    cudaMalloc(&d_c, count * sizeof(float));

    cudaMemcpy(d_a, a.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    NaiveGemmKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, count * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
