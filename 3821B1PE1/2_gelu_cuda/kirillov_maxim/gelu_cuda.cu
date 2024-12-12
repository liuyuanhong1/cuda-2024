// Copyright (c) 2024 Kirillov Maxim
#include <math.h>
#include <iostream>

#include "gelu_cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void GeluKernel(const float* input, float* output, size_t size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }
    const float geluParameter = 0.044715f;
    const float piParameter = sqrtf(2.0f / M_PI);

    float x = input[index];
    output[index] = 0.5f * x * (1.0f + tanhf(piParameter * (x + geluParameter * x * x * x)));
}


std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }
    size_t size = input.size();

    std::vector<float> output(size);
    float* deviceInput = nullptr;
    float* deviceOutput = nullptr;
    size_t sizeInBytes = size * sizeof(float);

    cudaMalloc(&deviceInput, sizeInBytes);
    cudaMalloc(&deviceOutput, sizeInBytes);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    cudaMemcpy(deviceInput, input.data(), sizeInBytes, cudaMemcpyHostToDevice);

    auto threads_per_block = deviceProp.maxThreadsPerBlock;
    auto blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    GeluKernel<<<blocks_per_grid, threads_per_block >>>(deviceInput, deviceOutput, size);
    cudaMemcpy(output.data(), deviceOutput, sizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    return output;
}
