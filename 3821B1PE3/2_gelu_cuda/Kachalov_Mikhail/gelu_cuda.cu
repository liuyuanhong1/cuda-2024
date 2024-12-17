// Copyright 2024 Kachalov Mikhail

#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

__global__ void geluKernel(const float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        const float sqrt_pi = sqrtf(2.0f / M_PI);
        const float c = 0.044715f;
        float x = input[idx];
        float cube_term = c * x * x * x;
        float tanh_arg = sqrt_pi * (x + cube_term);
        float tanh_value = tanhf(tanh_arg);

        output[idx] = 0.5f * x * (1.0f + tanh_value);
    }
}

std::vector<float> GeluCUDA(const std::vector<float> &input)
{
    size_t size = input.size();
    size_t bytes = size * sizeof(float);
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, bytes);
    cudaMalloc((void **)&d_output, bytes);
    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    geluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);
    std::vector<float> output(size);
    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}