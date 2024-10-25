// Copyright (c) 2024 Volodin Evgeniy
#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>
#include <algorithm>

__global__ void gelu_kernel(const float* input, float* output, std::size_t size) {
    const float sqrt_2pi = sqrtf(2.0f / M_PI);
    const float coeff_cubic = 0.044715f;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanf(sqrt_2pi * (x + coeff_cubic * x * x * x)));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) {
        throw std::invalid_argument("Input vector is empty!");
    }

    cudaDeviceProp deviceProp;
    cudaError_t cudaErr = cudaGetDeviceProperties(&deviceProp, 0);
    if (cudaErr != cudaSuccess) {
        throw std::runtime_error("Failed to get device properties!");
    }

    std::size_t size = input.size();
    std::vector<float> output(size);

    float* ptr_input = nullptr;
    float* ptr_output = nullptr;


    cudaErr = cudaMalloc(&ptr_input, size * sizeof(float));
    if (cudaErr != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for input.");
    }

    cudaErr = cudaMalloc(&ptr_output, size * sizeof(float));
    if (cudaErr != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for output.");
    }

    cudaErr = cudaMemcpy(ptr_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        cudaFree(ptr_input);
        cudaFree(ptr_output);
        throw std::runtime_error("Failed to copy input data to device.");
    }

    int blockSize = std::min(deviceProp.maxThreadsPerBlock, 1024);
    int numBlocks = (size + blockSize - 1) / blockSize;

    gelu_kernel<<<numBlocks, blockSize>>>(ptr_input, ptr_output, size);

    cudaErr = cudaDeviceSynchronize();
    if (cudaErr != cudaSuccess) {
        cudaFree(ptr_input);
        cudaFree(ptr_output);
        throw std::runtime_error("CUDA kernel execution failed.");
    }

    cudaErr = cudaMemcpy(output.data(), ptr_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaErr != cudaSuccess) {
        cudaFree(ptr_input);
        cudaFree(ptr_output);
        throw std::runtime_error("Failed to copy output data to host.");
    }

    cudaFree(ptr_input);
    cudaFree(ptr_output);

    return output;
}
