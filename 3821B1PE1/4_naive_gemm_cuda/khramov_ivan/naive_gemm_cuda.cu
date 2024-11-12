// Copyright (c) 2024 Khramov Ivan
#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>


__global__ void NaiveGemmKernel(const float* a,
                       const float* b,
                       float* c,
                       int n) {

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
    int elemCount = n * n;
    if (a.size() != elemCount || b.size() != elemCount) return {};

    std::vector<float> c(elemCount, 0.0f);

    float* d_a;
    float* d_b;
    float* d_c;

    cudaMalloc(&d_a, n * n * sizeof(float));
    cudaMalloc(&d_b, n * n * sizeof(float));
    cudaMalloc(&d_c, n * n * sizeof(float));

    cudaMemcpy(d_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    const size_t axSize = 32u;
    dim3 threadsPerBlock(
        axSize,
        axSize
    );
    dim3 numBlocks(
        (n + axSize - 1) / axSize,
        (n + axSize - 1) / axSize
    );

    NaiveGemmKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, elemCount * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
