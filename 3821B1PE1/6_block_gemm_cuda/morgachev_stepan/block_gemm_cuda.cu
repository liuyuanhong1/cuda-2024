// Copyright (c) 2024 Morgachev Stepan
#include "block_gemm_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void blockGemmKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float blockA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float blockB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int blockIdxK = 0; blockIdxK < n / BLOCK_SIZE; ++blockIdxK) {
        if (row < n && blockIdxK * BLOCK_SIZE + threadIdx.x < n) {
            blockA[threadIdx.y][threadIdx.x] = a[row * n + blockIdxK * BLOCK_SIZE + threadIdx.x];
        } else {
            blockA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < n && blockIdxK * BLOCK_SIZE + threadIdx.y < n) {
            blockB[threadIdx.y][threadIdx.x] = b[(blockIdxK * BLOCK_SIZE + threadIdx.y) * n + col];
        } else {
            blockB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += blockA[threadIdx.y][k] * blockB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}


std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n){

    std::vector<float> output(n * n, 0.0f);
    size_t sizeInBytes = n * n * sizeof(float);

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceOutput = nullptr;

    cudaMalloc(&deviceA, sizeInBytes);
    cudaMalloc(&deviceB, sizeInBytes);
    cudaMalloc(&deviceOutput, sizeInBytes);

    cudaMemcpy(deviceA, a.data(), sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b.data(), sizeInBytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    blockGemmKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceOutput, n);

    cudaMemcpy(output.data(), deviceOutput, sizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceOutput);

    return output;
}