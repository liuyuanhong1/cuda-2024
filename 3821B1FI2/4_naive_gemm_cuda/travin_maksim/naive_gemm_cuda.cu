#include <cuda_runtime.h>
#include <iostream>
#include "naive_gemm_cuda.h"

#define BLOCK_SIZE 16

void CUDA_CHECK(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void naive_gemm_kernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    if (row < n && col < n) {
        for (int k = 0; k < n; ++k) {
            c[row * n + col] += a[row * n + k] * b[k * n + col];
        }
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t bytes = n * n * sizeof(float);
    std::vector<float> c(n * n, 0.0f);

    float *d_a, *d_b, *d_c;

    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    naive_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, n);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return c;
}