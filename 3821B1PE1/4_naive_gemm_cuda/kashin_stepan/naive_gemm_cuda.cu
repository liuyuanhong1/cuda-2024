// Copyright (c) 2024 Kashin Stepan

#include <cstdlib>
#include <iostream>
#include <chrono>
#include <thread>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "naive_gemm_cuda.h"

#define BLOCK_DIM 32

__global__ void MatrixMultiplyKernel(const float* matrixA, const float* matrixB, float* matrixC,
                                     const size_t matrixSize)
{
    constexpr auto blockDim = BLOCK_DIM;
    __shared__ float sharedA[blockDim][blockDim];
    __shared__ float sharedB[blockDim][blockDim];

    size_t rowIdx = blockIdx.y * blockDim + threadIdx.y;
    size_t colIdx = blockIdx.x * blockDim + threadIdx.x;

    float tempResult = 0.0f;

    for (size_t k = 0; k < matrixSize; k += blockDim) {

        if (colIdx < matrixSize && (threadIdx.y + k) < matrixSize) {
            sharedB[threadIdx.y][threadIdx.x] = __ldg(&matrixB[(threadIdx.y + k) * matrixSize + colIdx]);
        } else {
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (rowIdx < matrixSize && (threadIdx.x + k) < matrixSize) {
            sharedA[threadIdx.y][threadIdx.x] = __ldg(&matrixA[rowIdx * matrixSize + threadIdx.x + k]);
        } else {
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (size_t l = 0; l < blockDim; ++l) {
            tempResult += sharedA[threadIdx.y][l] * sharedB[l][threadIdx.x];
        }

        __syncthreads();
    }

    if (rowIdx < matrixSize && colIdx < matrixSize) {
        matrixC[rowIdx * matrixSize + colIdx] = tempResult;
    }
}

std::vector<float> PerformMatrixMultiplicationCUDA(const std::vector<float>& matrixA,
                                                  const std::vector<float>& matrixB, int matrixSize) {
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);

    auto totalElements = matrixSize * matrixSize;
    std::vector<float> result(totalElements);
    auto totalSizeInBytes = totalElements * sizeof(float);

    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
    auto blocksPerGrid = (matrixSize + BLOCK_DIM - 1) / BLOCK_DIM;
    dim3 numBlocks(blocksPerGrid, blocksPerGrid);

    float *aDevice = nullptr;
    cudaMalloc(&aDevice, totalSizeInBytes);

    float *bDevice = nullptr;
    cudaMalloc(&bDevice, totalSizeInBytes);

    float *cDevice = nullptr;
    cudaMalloc(&cDevice, totalSizeInBytes);

    cudaMemcpy(aDevice, matrixA.data(), totalSizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bDevice, matrixB.data(), totalSizeInBytes, cudaMemcpyHostToDevice);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    MatrixMultiplyKernel<<<numBlocks, threadsPerBlock>>>(aDevice, bDevice, cDevice, matrixSize);

    cudaDeviceSynchronize();
    cudaMemcpy(result.data(), cDevice, totalSizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(cDevice);
    cudaFree(bDevice);
    cudaFree(aDevice);

    return result;
}
