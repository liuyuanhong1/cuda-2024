// Copyright (c) 2024 Nedelin Dmitry

#include <cuda_runtime.h>
#include "block_gemm_cuda.h"

#define TILE_SIZE 16

__global__ void BlockGemmKernel(const float* matrixA, const float* matrixB,
                                float* matrixC, int dimension) {
    __shared__ float sharedTileA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedTileB[TILE_SIZE][TILE_SIZE];

    int globalRow = blockIdx.y * TILE_SIZE + threadIdx.y;
    int globalCol = blockIdx.x * TILE_SIZE + threadIdx.x;

    float partialSum = 0.0f;

    for (int tileIdx = 0; tileIdx < dimension / TILE_SIZE; ++tileIdx) {
        sharedTileA[threadIdx.y][threadIdx.x] = matrixA[globalRow * dimension + (tileIdx * TILE_SIZE + threadIdx.x)];
        sharedTileB[threadIdx.y][threadIdx.x] = matrixB[(tileIdx * TILE_SIZE + threadIdx.y) * dimension + globalCol];

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            partialSum += sharedTileA[threadIdx.y][k] * sharedTileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (globalRow < dimension && globalCol < dimension) {
        matrixC[globalRow * dimension + globalCol] = partialSum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& hostMatrixA,
                                 const std::vector<float>& hostMatrixB,
                                 int dimension) {
    std::vector<float> hostMatrixC(dimension * dimension, 0.0f);

    float* deviceMatrixA = nullptr;
    float* deviceMatrixB = nullptr;
    float* deviceMatrixC = nullptr;

    size_t matrixBytes = dimension * dimension * sizeof(float);

    cudaMalloc(&deviceMatrixA, matrixBytes);
    cudaMalloc(&deviceMatrixB, matrixBytes);
    cudaMalloc(&deviceMatrixC, matrixBytes);

    cudaMemcpy(deviceMatrixA, hostMatrixA.data(), matrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, hostMatrixB.data(), matrixBytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((dimension + TILE_SIZE - 1) / TILE_SIZE, (dimension + TILE_SIZE - 1) / TILE_SIZE);

    BlockGemmKernel<<<numBlocks, threadsPerBlock>>>(deviceMatrixA, deviceMatrixB, deviceMatrixC, dimension);

    cudaMemcpy(hostMatrixC.data(), deviceMatrixC, matrixBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceMatrixC);

    return hostMatrixC;
}
