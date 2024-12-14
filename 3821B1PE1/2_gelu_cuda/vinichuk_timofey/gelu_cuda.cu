// Copyright (c) 2024 Vinichuk Timofey

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gelu_cuda.h"

__global__ void geluKernel(const float* input, float* output, size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < count) {
        constexpr float geluCoef1 = 0.044715f;
        constexpr float geluCoef2 = 0.7978845608f;
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + tanhf(geluCoef2 * x * (1.0f + geluCoef1 * x * x)));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // получаем свойства device

    if (input.empty()) return {};

    auto size = input.size();

    size_t countBytes = size * sizeof(float);
    std::vector<float> output(size);



    float* input_block = nullptr;
    float* output_block = nullptr;

    cudaMalloc(&input_block, size * sizeof(float));
    cudaMalloc(&output_block, size * sizeof(float));
    cudaMemcpy(input_block, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    auto blockSize = deviceProp.maxThreadsPerBlock;
    auto numBlocks = (size + blockSize - 1) / blockSize;

    geluKernel << <numBlocks, blockSize >> > (input_block, output_block, size);
    cudaMemcpy(output.data(), output_block, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(input_block);
    cudaFree(output_block);

    return output;
}
