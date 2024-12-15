// Copyright (c) 2024 Smirnova Daria
#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdlib>
#include <cmath>

__global__ void geluKernel(const float* input, float* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float x = input[idx];
        result[idx] = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t vectorSize = input.size();
    std::vector<float> result(vectorSize);
    size_t sizeInBytes = vectorSize * sizeof(*input.data());

    float* dInput;
    float* dOutput;
    cudaMalloc(&dInput, sizeInBytes);
    cudaMalloc(&dOutput, sizeInBytes);

    cudaMemcpy(dInput, input.data(), sizeInBytes, cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, 0);
    size_t threadsBlock = deviceProps.maxThreadsPerBlock;
    size_t blocksGrid = (vectorSize + threadsBlock - 1) / threadsBlock;

    geluKernel<<<blocksGrid, threadsBlock>>>(dInput, dOutput, vectorSize);
    cudaMemcpy(result.data(), dOutput, sizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(dInput);
    cudaFree(dOutput);
    
    return result;
}