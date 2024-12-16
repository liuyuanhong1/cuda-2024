// Copyright (c) 2024 Moiseev Nikita
#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdlib>
#include <cmath>

__global__ void geluKernel(const float* input, float* output, size_t data_size) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const float gelu_coefficient = 0.044715f;

    if (thread_index < data_size) {
        float value = input[thread_index];
        output[thread_index] = 0.5f * value * (1.0f + tanh(sqrt(2.0f / M_PI)
            * (value + gelu_coefficient * value * value * value)));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t input_size = input.size();
    std::vector<float> output(input_size);
    size_t memory_size = input_size * sizeof(float);

    float* device_input;
    float* device_output;
    cudaMalloc(&device_input, memory_size);
    cudaMalloc(&device_output, memory_size);

    cudaMemcpy(device_input, input.data(), memory_size, cudaMemcpyHostToDevice);

    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);
    size_t threads_per_block = device_properties.maxThreadsPerBlock;
    size_t blocks_per_grid = (input_size + threads_per_block - 1) / threads_per_block;

    geluKernel<<<blocks_per_grid, threads_per_block>>>(device_input, device_output, input_size);
    cudaMemcpy(output.data(), device_output, memory_size, cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return output;
}