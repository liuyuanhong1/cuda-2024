// Copyright (c) 2024 Tushentsova Karina
#include "naive_gemm_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define SIZE_BLOCK 32

__global__ void NaiveGemmKernel(const float* a, const float* b, float* output, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float result = 0.0f;
        for (int k = 0; k < n; ++k) {
            result += a[row * n + k] * b[k * n + col];
        }
        output[row * n + col] = result;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n){
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

    NaiveGemmKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceOutput, n);
    cudaMemcpy(output.data(), deviceOutput, sizeBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceOutput);

    return output;
}