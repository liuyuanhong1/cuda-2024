// Copyright (c) 2024 Lebedeva Ksenia
#include "block_gemm_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <vector>

#define BLOCK_SIZE 32

__global__ void myKernel(const float* a, const float* b,
                         float* const c, const int size) {
    __shared__ float aCached[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float bCached[BLOCK_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int nIdx = blockIdx.y * BLOCK_SIZE + ty;
    const int mIdx = blockIdx.x * BLOCK_SIZE + tx;

    float cVal = 0.0f;

    for (int t = 0; t < size / BLOCK_SIZE; ++t) {
        aCached[ty][tx] = a[nIdx * size + t * BLOCK_SIZE + tx];
        bCached[ty][tx] = b[(t * BLOCK_SIZE + ty) * size + mIdx];

        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            cVal += aCached[ty][k] * bCached[k][tx];
        }
        __syncthreads();
    }

    if (nIdx < size && mIdx < size) {
        c[nIdx * size + mIdx] = cVal;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, int size) {
    std::vector<float> c(size * size);

    size_t sizeInBytes = size * size * sizeof(*a.data());

    float* d_a;
    cudaMalloc(&d_a, sizeInBytes);
    float* d_b;
    cudaMalloc(&d_b, sizeInBytes);
    float* d_c;
    cudaMalloc(&d_c, sizeInBytes);

    cudaMemcpy(d_a, a.data(), sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeInBytes, cudaMemcpyHostToDevice);

    const int sizeAxis = BLOCK_SIZE;
    dim3 threadsPerBlock(
        sizeAxis,
        sizeAxis);
    dim3 numBlocks(
        (size + sizeAxis - 1) / sizeAxis,
        (size + sizeAxis - 1) / sizeAxis);

    myKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, size);

    cudaMemcpy(c.data(), d_c, sizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return c;
}
