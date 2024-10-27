#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 32

__global__ void blockGemmKernel(const float* A, const float* B, float* C, int n) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    int cIndex = (blockRow * BLOCK_SIZE + row) * n + (blockCol * BLOCK_SIZE + col);
    float cValue = 0.0f;

    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    int numBlocks = n / BLOCK_SIZE;

    for (int m = 0; m < numBlocks; ++m) {
        sharedA[row][col] = A[(blockRow * BLOCK_SIZE + row) * n + (m * BLOCK_SIZE + col)];
        sharedB[row][col] = B[(m * BLOCK_SIZE + row) * n + (blockCol * BLOCK_SIZE + col)];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            cValue += sharedA[row][k] * sharedB[k][col];
        }

        __syncthreads();
    }

    C[cIndex] = cValue;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    size_t matrixSize = n * n * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_B, matrixSize);
    cudaMalloc((void**)&d_C, matrixSize);

    cudaMemcpy(d_A, a.data(), matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), matrixSize, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(n / BLOCK_SIZE, n / BLOCK_SIZE);

    blockGemmKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_C, matrixSize, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return c;
}
