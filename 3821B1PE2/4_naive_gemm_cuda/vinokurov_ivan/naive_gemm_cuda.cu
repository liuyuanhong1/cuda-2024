//  Copyright (c) 2024 Vinokurov Ivan

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "naive_gemm_cuda.h"

#define TILE_SIZE 32

__global__ void NaiveGemmKernel(const float* a,
    const float* b,
    float* c, int n) {
    int currentRow = blockIdx.y * blockDim.y + threadIdx.y;
    int currentCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (currentRow < n && currentCol < n) {
        float result = 0.0f;
        for (int idx = 0; idx < n; ++idx) {
            result += a[currentRow * n + idx] * b[idx * n + currentCol];
        }
        c[currentRow * n + currentCol] = result;
    }
}


std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    std::vector<float> c(n * n, 0.0f);

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
    dim3 numberOfBlocks((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    NaiveGemmKernel << <numberOfBlocks, threadsPerBlock >> > (deviceMtxA, deviceMtxB, deviceMtxC, n);
    cudaMemcpy(c.data(), deviceMtxC, sizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceMtxA);
    cudaFree(deviceMtxB);
    cudaFree(deviceMtxC);

    return c;
}