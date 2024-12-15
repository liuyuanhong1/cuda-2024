// Copyright (c) 2024 Moiseev Nikita
#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void BlockGemmKernel(const float* matrix_a, const float* matrix_b, float* matrix_c, int matrix_size, int block_size) {
    __shared__ float shared_a[16][16];
    __shared__ float shared_b[16][16];

    int row = blockIdx.y * block_size + threadIdx.y;
    int col = blockIdx.x * block_size + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < matrix_size; k += block_size) {
        if (row < matrix_size && k + threadIdx.x < matrix_size) {
            shared_a[threadIdx.y][threadIdx.x] = matrix_a[row * matrix_size + k + threadIdx.x];
        } else {
            shared_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < matrix_size && k + threadIdx.y < matrix_size) {
            shared_b[threadIdx.y][threadIdx.x] = matrix_b[(k + threadIdx.y) * matrix_size + col];
        } else {
            shared_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < block_size; ++i) {
            sum += shared_a[threadIdx.y][i] * shared_b[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < matrix_size && col < matrix_size) {
        matrix_c[row * matrix_size + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& matrix_a, const std::vector<float>& matrix_b, int matrix_size) {
    auto total_elements = matrix_size * matrix_size;
    if (matrix_a.size() != total_elements || matrix_b.size() != total_elements) return {};

    std::vector<float> matrix_c(total_elements, 0.0f);

    float* d_matrix_a;
    float* d_matrix_b;
    float* d_matrix_c;
    cudaMalloc(&d_matrix_a, total_elements * sizeof(float));
    cudaMalloc(&d_matrix_b, total_elements * sizeof(float));
    cudaMalloc(&d_matrix_c, total_elements * sizeof(float));

    cudaMemcpy(d_matrix_a, matrix_a.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, matrix_b.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 16;
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid((matrix_size + block_size - 1) / block_size, (matrix_size + block_size - 1) / block_size);

    BlockGemmKernel<<<dimGrid, dimBlock>>>(d_matrix_a, d_matrix_b, d_matrix_c, matrix_size, block_size);
    cudaMemcpy(matrix_c.data(), d_matrix_c, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_matrix_c);

    return matrix_c;
}
