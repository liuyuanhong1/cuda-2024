// Copyright (c) 2024 Korablev Nikita
#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

__global__ void BlockGemmKernel(const float* a, const float* b, float* c, int n, int blockSize) {
    __shared__ float aBlock[16][16];
    __shared__ float bBlock[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tmp = 0;

    for (int k = 0; k < n; k += blockSize) {
        if (row < n && k + threadIdx.y < n) {
            aBlock[threadIdx.y][threadIdx.x] = a[row * n + k + threadIdx.x];
        } else {
            aBlock[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < n && k + threadIdx.x < n) {
            bBlock[threadIdx.y][threadIdx.x] = b[(k + threadIdx.y) * n + col];
        } else {
            bBlock[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < blockSize; ++i) {
            tmp += aBlock[threadIdx.y][i] * bBlock[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = tmp;
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

    const int blockSize = 16;
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);

    BlockGemmKernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, n, blockSize);
    cudaMemcpy(c.data(), d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
