// Copyright (c) 2024 Morgachev Stepan
#include "naive_gemm_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void NaiveGemmKernel(const float* a, const float* b, float* output, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float result = 0.0f;
        for (int k = 0; k < n; ++k) {
            result = fma(a[row * n + k], b[k * n + col], result);
        }
        output[row * n + col] = result;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n){
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

    NaiveGemmKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceOutput, n);

    cudaMemcpy(output.data(), deviceOutput, sizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceOutput);

    return output;
}
