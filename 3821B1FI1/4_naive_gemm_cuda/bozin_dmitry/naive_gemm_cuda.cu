// Copyright (c) 2024 Bozin Dmitry
#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>



__global__ void MulMatrixKernel(const float* a, const float* b, float* c, int n) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (size_t k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}


std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    
    const size_t count = n * n;
    std::vector<float> result(count, 0.0f);
    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceResult = nullptr;
    cudaMalloc((void**)&deviceA, count * sizeof(float));
    cudaMalloc((void**)&deviceB, count * sizeof(float));
    cudaMalloc((void**)&deviceResult, count * sizeof(float));
    cudaMemcpy(deviceA, a.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device);
    const double block_dim_x = std::sqrt(device_prop.maxThreadsPerBlock);
    const double num_blocks_x = (n + block_dim_x - 1) / block_dim_x;
    dim3 blockSize(block_dim_x, block_dim_x);
    dim3 numBlocks(num_blocks_x, num_blocks_x);
    MulMatrixKernel<<<numBlocks, blockSize>>>(deviceA, deviceB, deviceResult, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMemcpy(result.data(), deviceResult, count * sizeof(float), cudaMemcpyDeviceToHost);

    return result;

}