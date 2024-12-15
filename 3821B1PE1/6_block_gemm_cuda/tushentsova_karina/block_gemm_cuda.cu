// Copyright (c) 2024 Tushentsova Karina
#include "block_gemm_cuda.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define SIZE_BLOCK 16

__global__ void blockGemmKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float blockA[SIZE_BLOCK][SIZE_BLOCK];
    __shared__ float blockB[SIZE_BLOCK][SIZE_BLOCK];

    int row = blockIdx.y * SIZE_BLOCK + threadIdx.y;
    int col = blockIdx.x * SIZE_BLOCK + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < n / SIZE_BLOCK; ++k) {
        if (row < n && k * SIZE_BLOCK + threadIdx.x < n) {
            blockA[threadIdx.y][threadIdx.x] = a[row * n + k * SIZE_BLOCK + threadIdx.x];
        } else {
            blockA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < n && k * SIZE_BLOCK + threadIdx.y < n) {
            blockB[threadIdx.y][threadIdx.x] = b[(k * SIZE_BLOCK + threadIdx.y) * n + col];
        } else {
            blockB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < SIZE_BLOCK; ++i) {
            sum += blockA[threadIdx.y][i] * blockB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}


std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    int totalElements = n * n;
    std::vector<float> output(totalElements, 0.0f);
    size_t sizeBytes = totalElements * sizeof(float);

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceOutput = nullptr;

    cudaMalloc(&deviceA, sizeBytes);
    cudaMalloc(&deviceB, sizeBytes);
    cudaMalloc(&deviceOutput, sizeBytes);

    cudaMemcpy(deviceA, a.data(), sizeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b.data(), sizeBytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(SIZE_BLOCK, SIZE_BLOCK);
    dim3 blocksPerGrid((n + SIZE_BLOCK - 1) / SIZE_BLOCK, (n + SIZE_BLOCK - 1) / SIZE_BLOCK);

    blockGemmKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceOutput, n);
    cudaMemcpy(output.data(), deviceOutput, sizeBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceOutput);

    return output;
}