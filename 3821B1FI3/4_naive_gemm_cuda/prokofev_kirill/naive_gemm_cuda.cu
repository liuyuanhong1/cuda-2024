// Copyright (c) 2024 Prokofev Kirill
#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>



__global__ void MulMatrixKernel(const float* a, const float* b, float* c, int n, int offset) {
    size_t row = offset + blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (size_t k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    
    
    const size_t count = n * n;
    std::vector<float> result(count, 0.0f);
    
    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceResult = nullptr;
    cudaMalloc(&deviceA, count * sizeof(float));
    cudaMalloc(&deviceB, count * sizeof(float));
    cudaMalloc(&deviceResult, count * sizeof(float));

   
    const int numStreams = 4; 
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    cudaMemcpy(deviceA, a.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    for (int i = 0; i < numStreams; ++i) {
        int offset = (n / numStreams) * i; 
        int rows = (i == numStreams - 1) ? n - offset : (n / numStreams); 

        MulMatrixKernel<<<gridSize, blockSize, 0, streams[i]>>>(deviceA, deviceB, deviceResult, rows, offset);
    }

    cudaMemcpy(result.data(), deviceResult, count * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numStreams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return result;

}
