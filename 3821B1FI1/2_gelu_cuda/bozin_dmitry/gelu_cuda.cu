// Copyright (c) 2024 Bozin Dmitry
#include "gelu_cuda.h"
#include <cmath>
#include <cuda_runtime.h>
#include<iostream>
#include<cstdlib>
#include<random>
#include<chrono>


__global__ void GelKern(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr float sqrt2_pi = 0.797885f;
    constexpr float coeff = 0.044715f;
    if (idx < size) {
        float x = input[idx]; 
        float tanh_arg = sqrt2_pi * (x + coeff * x * x * x);
        output[idx] = 0.5f * x * (1.0f + tanh(tanh_arg));
    }
}



std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t vectorSize = input.size();
    const int blockSize = 256;
    int numBlocks = (vectorSize + blockSize - 1) / blockSize;
    std::vector<float>result(vectorSize);
    float* deviceInput = nullptr;
    float* deviceOutput = nullptr;
    cudaMalloc(&deviceInput,vectorSize * sizeof(float));
    cudaMalloc(&deviceOutput,vectorSize * sizeof(float));
    cudaMemcpy(deviceInput, input.data(), vectorSize * sizeof(float), cudaMemcpyHostToDevice);
    GelKern<<<numBlocks, blockSize>>>(deviceInput, deviceOutput, vectorSize);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMemcpy(result.data(), deviceOutput, vectorSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    return result;
}
