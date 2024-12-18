// Copyright 2024 Kachalov Mikhail
#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16

__global__ void MatrixMulKernel(const float *A, const float *B, float *C, int n)
{
    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float cValue = 0.0f;

    for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++k)
    {
        if (row < n && (k * BLOCK_SIZE + tx) < n)
            Asub[ty][tx] = A[row * n + (k * BLOCK_SIZE + tx)];
        else
            Asub[ty][tx] = 0.0f;

        if ((k * BLOCK_SIZE + ty) < n && col < n)
            Bsub[ty][tx] = B[(k * BLOCK_SIZE + ty) * n + col];
        else
            Bsub[ty][tx] = 0.0f;

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e)
        {
            cValue += Asub[ty][e] * Bsub[e][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n)
    {
        C[row * n + col] = cValue;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 int n)
{
    size_t size = n * n * sizeof(float);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    MatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}