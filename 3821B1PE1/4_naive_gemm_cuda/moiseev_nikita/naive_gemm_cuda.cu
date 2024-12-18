// Copyright (c) 2024 Moiseev Nikita
#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void NaiveGemmKernel(const float* matrix_a, const float* matrix_b, float* matrix_c, int dimension) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < dimension && col < dimension) {
        float element_sum = 0.0f;
        for (int k = 0; k < dimension; ++k) {
            element_sum += matrix_a[row * dimension + k] * matrix_b[k * dimension + col];
        }
        matrix_c[row * dimension + col] = element_sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& matrix_a, const std::vector<float>& matrix_b, int dimension) {
    int total_elements = dimension * dimension;
    if (matrix_a.size() != total_elements || matrix_b.size() != total_elements) {
        return {};
    }

    std::vector<float> result_matrix(total_elements, 0.0f);

    float* device_matrix_a;
    float* device_matrix_b;
    float* device_matrix_c;

    cudaMalloc(&device_matrix_a, total_elements * sizeof(float));
    cudaMalloc(&device_matrix_b, total_elements * sizeof(float));
    cudaMalloc(&device_matrix_c, total_elements * sizeof(float));

    cudaMemcpy(device_matrix_a, matrix_a.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrix_b, matrix_b.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);

    const size_t block_size = 32u;
    dim3 threads_per_block(block_size, block_size);
    dim3 num_blocks((dimension + block_size - 1) / block_size, (dimension + block_size - 1) / block_size);

    NaiveGemmKernel<<<num_blocks, threads_per_block>>>(device_matrix_a, device_matrix_b, device_matrix_c, dimension);

    cudaMemcpy(result_matrix.data(), device_matrix_c, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_matrix_a);
    cudaFree(device_matrix_b);
    cudaFree(device_matrix_c);

    return result_matrix;
}