// Copyright 2024 Kachalov Mikhail
#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void NaiveGemmKernel(const float *a, const float *b, float *c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float value = 0.0f;
        for (int k = 0; k < n; ++k)
        {
            value += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = value;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 int n)
{
    int blockSize = 16;
    dim3 block(blockSize, blockSize);
    dim3 grid((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * n * sizeof(float));
    cudaMalloc(&d_b, n * n * sizeof(float));
    cudaMalloc(&d_c, n * n * sizeof(float));
    cudaMemcpy(d_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    NaiveGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}