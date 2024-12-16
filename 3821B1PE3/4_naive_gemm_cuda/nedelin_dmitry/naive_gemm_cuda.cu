// Copyright (c) 2024 Nedelin Dmitry

#include <cuda_runtime.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "naive_gemm_cuda.h"

#define TILE_SIZE 32

__global__ void NaiveGemmKernel(const float* matrixA,
    const float* matrixB,
    float* matrixC, int dimension) {
    int currentRow = blockIdx.y * blockDim.y + threadIdx.y;
    int currentCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (currentRow < dimension && currentCol < dimension) {
        float result = 0.0f;
        for (int idx = 0; idx < dimension; ++idx) {
            result += matrixA[currentRow * dimension + idx] * matrixB[idx * dimension + currentCol];
        }
        matrixC[currentRow * dimension + currentCol] = result;
    }
}


std::vector<float> NaiveGemmCUDA(const std::vector<float>& matrixA,
    const std::vector<float>& matrixB,
    int dimension) {
    std::vector<float> matrixC(dimension * dimension, 0.0f);

    float* deviceMatrixA = nullptr;
    float* deviceMatrixB = nullptr;
    float* deviceMatrixC = nullptr;

    size_t totalSizeInBytes = dimension * dimension * sizeof(float);

    cudaMalloc(&deviceMatrixA, totalSizeInBytes);
    cudaMalloc(&deviceMatrixB, totalSizeInBytes);
    cudaMalloc(&deviceMatrixC, totalSizeInBytes);

    cudaMemcpy(deviceMatrixA, matrixA.data(), totalSizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, matrixB.data(), totalSizeInBytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numberOfBlocks((dimension + TILE_SIZE - 1) / TILE_SIZE, (dimension + TILE_SIZE - 1) / TILE_SIZE);

    NaiveGemmKernel << <numberOfBlocks, threadsPerBlock >> > (deviceMatrixA, deviceMatrixB, deviceMatrixC, dimension);
    cudaMemcpy(matrixC.data(), deviceMatrixC, totalSizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceMatrixC);

    return matrixC;
}
