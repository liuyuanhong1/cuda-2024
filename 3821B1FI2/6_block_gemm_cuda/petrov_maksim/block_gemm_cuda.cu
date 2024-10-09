#include <cuda_runtime.h>
#include <iostream>
#include "block_gemm_cuda.h"

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

const int TILE_SIZE = 32;

__global__ void blockGemmKernel(const float* A, const float* B, float* C, int n) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    float sum = 0.0f;

    for (int m = 0; m < (n + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        
        if ((blockRow * TILE_SIZE + threadIdx.y < n) && (m * TILE_SIZE + threadIdx.x < n)) {
            tile_A[threadIdx.y][threadIdx.x] = A[(blockRow * TILE_SIZE + threadIdx.y) * n + (m * TILE_SIZE + threadIdx.x)];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((m * TILE_SIZE + threadIdx.y < n) && (blockCol * TILE_SIZE + threadIdx.x < n)) {
            tile_B[threadIdx.y][threadIdx.x] = B[(m * TILE_SIZE + threadIdx.y) * n + (blockCol * TILE_SIZE + threadIdx.x)];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    int row = blockRow * TILE_SIZE + threadIdx.y;
    int col = blockCol * TILE_SIZE + threadIdx.x;
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}


std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    size_t size = n * n * sizeof(float);

    float* d_a;
    float* d_b;
    float* d_c;

    checkCudaError(cudaMalloc((void**)&d_a, size), "Failed to allocate memory for d_a");
    checkCudaError(cudaMalloc((void**)&d_b, size), "Failed to allocate memory for d_b");
    checkCudaError(cudaMalloc((void**)&d_c, size), "Failed to allocate memory for d_c");

    checkCudaError(cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice), "Failed to copy data for d_a");
    checkCudaError(cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice), "Failed to copy data for d_b");

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    blockGemmKernel << <numBlocks, threadsPerBlock >> > (d_a, d_b, d_c, n);
    checkCudaError(cudaGetLastError(), "Error when starting the kernel");

    checkCudaError(cudaDeviceSynchronize(), "Synchronization error");

    std::vector<float> c(n * n);
    checkCudaError(cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost), "Failed to copy data for c");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}