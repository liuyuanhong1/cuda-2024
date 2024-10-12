// Copyright (c) 2024 Prokofev Kirill
#include "gelu_cuda.h"
#include <cmath>
#include <cuda_runtime.h>
#include<iostream>
#include<cstdlib>
#include<random>
#include<chrono>

__device__ float fast_tanh(float x) {
    if (x < -3.0f) return -1.0f;
    if (x > 3.0f) return 1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}


/*
std::vector<float> GenVec(const int size){
    std::vector<float> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for(size_t i = 0;i < vec.size();++i){
        vec[i] = dist(gen);
    }
    return vec;
}
*/


__global__ void GelKern(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr float sqrt2_pi = 0.797885f;
    constexpr float coeff = 0.044715f;
    if (idx < size) {
        float x = input[idx]; 
        float tanh_arg = sqrt2_pi * (x + coeff * x * x * x);
        output[idx] = 0.5f * x * (1.0f + fast_tanh(tanh_arg));
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


/*
int main() {
    
    //std::vector<float> a {0.1,2.4,5.4,3.2};
    std::vector<float> a = GenVec(134217728);
    
    // Performance Measuring
    auto start = std::chrono::high_resolution_clock::now();
    auto c = GeluCUDA(a);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " s" << std::endl;

    
    return 0;
}
*/