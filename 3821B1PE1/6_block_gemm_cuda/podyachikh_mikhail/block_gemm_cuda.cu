// Copyright (c) 2024 Podyachikh Mikhail
#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

static const int block_size = 16;

__global__ void BlockGemmKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float shared_a[block_size][block_size];
    __shared__ float shared_b[block_size][block_size];

    int row = blockIdx.y * block_size + threadIdx.y;
    int col = blockIdx.x * block_size + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < n; k += block_size) {
        if (row < n && k + threadIdx.x < n) {
            shared_a[threadIdx.y][threadIdx.x] = a[row * n + k + threadIdx.x];
        } else {
            shared_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < n && k + threadIdx.y < n) {
            shared_b[threadIdx.y][threadIdx.x] = b[(k + threadIdx.y) * n + col];
        } else {
            shared_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < block_size; ++i) {
            sum += shared_a[threadIdx.y][i] * shared_b[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    auto total_elements = n * n;
    std::vector<float> c(total_elements);

    float* buf_a, * buf_b, * buf_c;
    cudaMalloc(&buf_a, total_elements * sizeof(float));
    cudaMalloc(&buf_b, total_elements * sizeof(float));
    cudaMalloc(&buf_c, total_elements * sizeof(float));

    cudaMemcpy(buf_a, a.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(buf_b, b.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);

    BlockGemmKernel<<<dimGrid, dimBlock>>>(buf_a, buf_b, buf_c, n);
    cudaMemcpy(c.data(), buf_c, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(buf_a);
    cudaFree(buf_b);
    cudaFree(buf_c);

    return c;
}
