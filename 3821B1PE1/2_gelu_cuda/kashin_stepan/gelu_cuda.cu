// Copyright (c) 2024 Kashin Stepan

#include <vector>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gelu_cuda.h"

__global__ void ApplyGelu(const float* input, float* output, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length) {
        constexpr float coeff1 = 1.595769122f;
        constexpr float coeff2 = 0.071354816f;

        float x = input[idx];
        output[idx] = x * (1 - 1 / (1.0f + __expf(x * (coeff1 + x * x * coeff2))));
    }
}

std::vector<float> ComputeGeluCUDA(const std::vector<float>& input) {
    if (input.empty()) return {};

    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, 0);

    size_t length = input.size();
    std::vector<float> output(length);

    size_t bytes = length * sizeof(float);
    int threads = deviceProps.maxThreadsPerBlock;
    int blocks = (length + threads - 1) / threads;

    float* devInput = nullptr;
    float* devOutput = nullptr;
    cudaMalloc(&devInput, bytes);
    cudaMalloc(&devOutput, bytes);

    cudaMemcpy(devInput, input.data(), bytes, cudaMemcpyHostToDevice);

    ApplyGelu<<<blocks, threads>>>(devInput, devOutput, length);

    cudaDeviceSynchronize();
    cudaMemcpy(output.data(), devOutput, bytes, cudaMemcpyDeviceToHost);

    cudaFree(devOutput);
    cudaFree(devInput);

    return output;
}