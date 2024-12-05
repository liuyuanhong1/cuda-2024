#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <iostream>

#define CUDA_CHECK_ERROR(call)                                            \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            throw std::runtime_error(std::string("CUDA Error: ") +        \
                                    cudaGetErrorString(err));             \
        }                                                                 \
    } while (0)

#define TILE_SIZE 16

__global__
void TiledGemmKernel(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C,
                     int n) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < n && t * TILE_SIZE + threadIdx.x < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < n && t * TILE_SIZE + threadIdx.y < n)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

std::vector<float> OptimizedGemmCUDA(const std::vector<float>& A,
                                     const std::vector<float>& B,
                                     int n) {
    if (A.size() != static_cast<size_t>(n * n) ||
        B.size() != static_cast<size_t>(n * n)) {
        throw std::invalid_argument("Input matrices must be of size n x n.");
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t bytes = n * n * sizeof(float);
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_C, bytes));

    CUDA_CHECK_ERROR(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    TiledGemmKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);

    CUDA_CHECK_ERROR(cudaGetLastError());

    std::vector<float> C(n * n);

    CUDA_CHECK_ERROR(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaFree(d_A));
    CUDA_CHECK_ERROR(cudaFree(d_B));
    CUDA_CHECK_ERROR(cudaFree(d_C));

    return C;
}
