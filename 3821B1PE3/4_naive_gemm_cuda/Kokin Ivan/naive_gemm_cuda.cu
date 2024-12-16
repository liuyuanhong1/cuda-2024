// Copyright (c) 2024 Kokin Ivan

#include <cuda_runtime.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "naive_gemm_cuda.h"

#define TILE_SIZE 32

__global__ void NaiveGemmKernel(const float* mxA,
    const float* mxB,
    float* mxC, int src) {
    int currR = blockIdx.y * blockDim.y + threadIdx.y;
    int currC = blockIdx.x * blockDim.x + threadIdx.x;

    if (currR < src && currC < src) {
        float res = 0.0f;
        for (int idx = 0; idx < src; ++idx) {
            res += mxA[currR * src + idx] * mxB[idx * src + currC];
        }
        mxC[currR * src + currC] = res;
    }
}


std::vector<float> NaiveGemmCUDA(const std::vector<float>& mxA,
    const std::vector<float>& mxB,
    int src) {
    std::vector<float> mxC(src * src, 0.0f);

    float* devMxA = nullptr;
    float* devMxB = nullptr;
    float* devMxC = nullptr;

    size_t totalSizeInBytes = src * src * sizeof(float);

    cudaMalloc(&devMxA, totalSizeInBytes);
    cudaMalloc(&devMxB, totalSizeInBytes);
    cudaMalloc(&devMxC, totalSizeInBytes);

    cudaMemcpy(devMxA, mxA.data(), totalSizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(devMxB, mxB.data(), totalSizeInBytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numberOfBlocks((src + TILE_SIZE - 1) / TILE_SIZE, (src + TILE_SIZE - 1) / TILE_SIZE);

    NaiveGemmKernel << <numberOfBlocks, threadsPerBlock >> > (devMxA, devMxB, devMxC, src);
    cudaMemcpy(mxC.data(), devMxC, totalSizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(devMxA);
    cudaFree(devMxB);
    cudaFree(devMxC);

    return mxC;
}
