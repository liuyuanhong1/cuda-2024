//  Copyright (c) 2024 Vinokurov Ivan
#include <cuda_runtime.h>
#include "block_gemm_cuda.h"

#define TILE_SIZE 16

__global__ void BlockGemmKernel(const float* a, const float* b,
                                float* c, int n) {
    __shared__ float sharedTileA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedTileB[TILE_SIZE][TILE_SIZE];

    int globalRow = blockIdx.y * TILE_SIZE + threadIdx.y;
    int globalCol = blockIdx.x * TILE_SIZE + threadIdx.x;

    float partialSum = 0.0f;

    for (int tileIdx = 0; tileIdx < n / TILE_SIZE; ++tileIdx) {
        sharedTileA[threadIdx.y][threadIdx.x] = a[globalRow * n + (tileIdx * TILE_SIZE + threadIdx.x)];
        sharedTileB[threadIdx.y][threadIdx.x] = b[(tileIdx * TILE_SIZE + threadIdx.y) * n + globalCol];

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            partialSum += sharedTileA[threadIdx.y][k] * sharedTileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (globalRow < n && globalCol < n) {
        c[globalRow * n + globalCol] = partialSum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> output(n * n, 0.0f);

    float* deviceMtxA = nullptr;
    float* deviceMtxB = nullptr;
    float* deviceMtxC = nullptr;

    size_t sizeInBytes = n * n * sizeof(float);

    cudaMalloc(&deviceMtxA, sizeInBytes);
    cudaMalloc(&deviceMtxB, sizeInBytes);
    cudaMalloc(&deviceMtxC, sizeInBytes);

    cudaMemcpy(deviceMtxA, a.data(), sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMtxB, b.data(), sizeInBytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    BlockGemmKernel<<<numBlocks, threadsPerBlock>>>(deviceMtxA, deviceMtxB, deviceMtxC, n);

    cudaMemcpy(output.data(), deviceMtxC, sizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceMtxA);
    cudaFree(deviceMtxB);
    cudaFree(deviceMtxC);

    return output;
}