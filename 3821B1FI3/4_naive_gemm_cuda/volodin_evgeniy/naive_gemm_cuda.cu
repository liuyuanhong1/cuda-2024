// Copyright (c) 2024 Volodin Evgeniy
#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <iostream>

__global__ void gemm_kernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (a.size() != n * n || b.size() != n * n) {
        throw std::invalid_argument("Matrix size does not match the specified n*n dimensions!");
    }

    std::vector<float> c(n * n, 0.0f);

    float *ptr_a, *ptr_b, *ptr_c;

    cudaError_t cudaErr = cudaMalloc(&ptr_a, n * n * sizeof(float));
    if (cudaErr != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for matrix A.");
    }

    cudaErr = cudaMalloc(&ptr_b, n * n * sizeof(float));
    if (cudaErr != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for matrix B.");
    }

    cudaErr = cudaMalloc(&ptr_c, n * n * sizeof(float));
    if (cudaErr != cudaSuccess) {
        cudaFree(ptr_a);
        cudaFree(ptr_b);
        throw std::runtime_error("Failed to allocate device memory for matrix C.");
    }

    cudaErr = cudaMemcpy(ptr_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        cudaFree(ptr_a);
        cudaFree(ptr_b);
        cudaFree(ptr_c);
        throw std::runtime_error("Failed to copy matrix A to device.");
    }

    cudaErr = cudaMemcpy(ptr_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess) {
        cudaFree(ptr_a);
        cudaFree(ptr_b);
        cudaFree(ptr_c);
        throw std::runtime_error("Failed to copy matrix B to device.");
    }

    int blockSize = 32;

    dim3 block(blockSize, blockSize);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    
    gemm_kernel<<<grid, block>>>(ptr_a, ptr_b, ptr_c, n);

    cudaDeviceSynchronize();

    cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        cudaFree(ptr_a);
        cudaFree(ptr_b);
        cudaFree(ptr_c);
        throw std::runtime_error("Kernel launch failed.");
    }

    cudaErr = cudaMemcpy(c.data(), ptr_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaErr != cudaSuccess) {
        cudaFree(ptr_a);
        cudaFree(ptr_b);
        cudaFree(ptr_c);
        throw std::runtime_error("Failed to copy result matrix C to host.");
    }

    cudaFree(ptr_a);
    cudaFree(ptr_b);
    cudaFree(ptr_c);

    return c;
}