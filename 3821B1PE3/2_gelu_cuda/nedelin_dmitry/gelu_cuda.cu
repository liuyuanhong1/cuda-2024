// Copyright 2024 Nedelin Dmitry

#define _USE_MATH_DEFINES

#include <math.h>
#include <iostream>

#include "gelu_cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void GeluKernel(const float* input_data, float* output_data, size_t num_elements) {
    size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx >= num_elements) {
        return;
    }
    const float GELU_COEFFICIENT = 0.044715f;
    const float PI_COEFFICIENT = sqrtf(2.0f / M_PI);

    float input_value = input_data[thread_idx];
    output_data[thread_idx] = 0.5f * input_value * (1.0f + tanhf(PI_COEFFICIENT * (input_value + GELU_COEFFICIENT * input_value * input_value * input_value)));
}

std::vector<float> GeluCUDA(const std::vector<float>& host_input) {
    size_t num_elements = host_input.size();

    std::vector<float> host_output(num_elements);
    float* device_input_data = nullptr;
    float* device_output_data = nullptr;
    size_t memory_size = num_elements * sizeof(float);

    cudaMalloc(&device_input_data, memory_size);
    cudaMalloc(&device_output_data, memory_size);

    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);

    cudaMemcpy(device_input_data, host_input.data(), memory_size, cudaMemcpyHostToDevice);

    auto threads_per_block = device_properties.maxThreadsPerBlock;
    auto blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    GeluKernel << <blocks_per_grid, threads_per_block >> > (device_input_data, device_output_data, num_elements);
    cudaMemcpy(host_output.data(), device_output_data, memory_size, cudaMemcpyDeviceToHost);

    cudaFree(device_input_data);
    cudaFree(device_output_data);

    return host_output;
}
