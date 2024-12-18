// Copyright (c) 2024 Sokolova Daria
#include "naive_gemm_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void NaiveGemmKernel(const float* matrixA, const float* matrixB, float* matrixC, int matrixDim) {
    int threadRow = blockIdx.y * blockDim.y + threadIdx.y;
    int threadCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadRow < matrixDim && threadCol < matrixDim) {
        float accum = 0.0f;
        for (int k = 0; k < matrixDim; ++k) {
            accum = fma(matrixA[threadRow * matrixDim + k], matrixB[k * matrixDim + threadCol], accum);
        }
        matrixC[threadRow * matrixDim + threadCol] = accum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    std::vector<float> resultMatrix(n * n, 0.0f);
    size_t totalBytes = n * n * sizeof(float);

    float* deviceMatrixA = nullptr;
    float* deviceMatrixB = nullptr;
    float* deviceMatrixC = nullptr;

    cudaMalloc(&deviceMatrixA, totalBytes);
    cudaMalloc(&deviceMatrixB, totalBytes);
    cudaMalloc(&deviceMatrixC, totalBytes);

    cudaMemcpy(deviceMatrixA, a.data(), totalBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, b.data(), totalBytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    NaiveGemmKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceMatrixA, deviceMatrixB, deviceMatrixC, n);

    cudaMemcpy(resultMatrix.data(), deviceMatrixC, totalBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceMatrixC);

    return resultMatrix;
}
