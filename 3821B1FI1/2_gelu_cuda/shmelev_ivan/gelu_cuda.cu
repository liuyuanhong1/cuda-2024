// Copyright (c) 2024 Shmelev Ivan
#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdlib>
#include <cmath>

__global__ void geluKernel(const float* input, float* result, size_t dataSize) {

    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const float geluConstant = 0.044715f;

    if (threadIndex < dataSize) {
        float x = input[threadIndex];
        result[threadIndex] = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI)
         * (x + geluConstant * x * x * x)));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t vectorSize = input.size();
    std::vector<float> result(vectorSize);
    size_t sizeInBytes = vectorSize * sizeof(*input.data());

    float* deviceInput;
    float* deviceOutput;
    cudaMalloc(&deviceInput, sizeInBytes);
    cudaMalloc(&deviceOutput, sizeInBytes);

    cudaMemcpy(deviceInput, input.data(), sizeInBytes, cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, 0);
    size_t threadsBlock = deviceProps.maxThreadsPerBlock;
    size_t blocksGrid = (vectorSize + threadsBlock - 1) / threadsBlock;

    geluKernel<<<blocksGrid, threadsBlock>>>(deviceInput, deviceOutput, vectorSize);
    cudaMemcpy(result.data(), deviceOutput, sizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    return result;
}