// Copyright (c) 2024 Sokolova Daria
#include "block_gemm_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void blockGemmKernel(
  const float* matrixA,
  const float* matrixB,
  float* matrixC,
  int matrixDim
  ) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int globalRow = blockIdx.y * TILE_SIZE + threadIdx.y;
    int globalCol = blockIdx.x * TILE_SIZE + threadIdx.x;

    float partialSum = 0.0f;

    for (int tileIdx = 0; tileIdx < matrixDim / TILE_SIZE; ++tileIdx) {
        if (globalRow < matrixDim && tileIdx * TILE_SIZE + threadIdx.x < matrixDim) {
            tileA[threadIdx.y][threadIdx.x] = matrixA[globalRow * matrixDim + tileIdx * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (globalCol < matrixDim && tileIdx * TILE_SIZE + threadIdx.y < matrixDim) {
            tileB[threadIdx.y][threadIdx.x] = matrixB[(tileIdx * TILE_SIZE + threadIdx.y) * matrixDim + globalCol];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            partialSum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (globalRow < matrixDim && globalCol < matrixDim) {
        matrixC[globalRow * matrixDim + globalCol] = partialSum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& matrixA,
                                 const std::vector<float>& matrixB,
                                 int matrixDim) {

    std::vector<float> resultMatrix(matrixDim * matrixDim, 0.0f);
    size_t totalBytes = matrixDim * matrixDim * sizeof(float);

    float* deviceMatrixA = nullptr;
    float* deviceMatrixB = nullptr;
    float* deviceMatrixC = nullptr;

    cudaMalloc(&deviceMatrixA, totalBytes);
    cudaMalloc(&deviceMatrixB, totalBytes);
    cudaMalloc(&deviceMatrixC, totalBytes);

    cudaMemcpy(deviceMatrixA, matrixA.data(), totalBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, matrixB.data(), totalBytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((matrixDim + TILE_SIZE - 1) / TILE_SIZE, (matrixDim + TILE_SIZE - 1) / TILE_SIZE);

    blockGemmKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceMatrixA, deviceMatrixB, deviceMatrixC, matrixDim);

    cudaMemcpy(resultMatrix.data(), deviceMatrixC, totalBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceMatrixC);

    return resultMatrix;
}
