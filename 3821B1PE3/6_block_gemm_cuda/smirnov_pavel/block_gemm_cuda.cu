#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

const int BLOCK_SIZE = 16;

__global__ void blockGemmKernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;

        __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

        for (int k = 0; k < n; k += BLOCK_SIZE) {
            if (row < n && k + threadIdx.y < n) {
                sharedA[threadIdx.y][threadIdx.x] = a[row * n + k + threadIdx.x];
            }
            if (col < n && k + threadIdx.x < n) {
                sharedB[threadIdx.y][threadIdx.x] = b[(k + threadIdx.y) * n + col];
            }

            __syncthreads();

            for (int i = 0; i < BLOCK_SIZE; ++i) {
                sum += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];
            }

            __syncthreads();
        }

        c[row * n + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, 
                                 const std::vector<float>& b,
                                 int n) {
    float* d_a;
    float* d_b;
    float* d_c;
    cudaMalloc(&d_a, n * n * sizeof(float));
    cudaMalloc(&d_b, n * n * sizeof(float));
    cudaMalloc(&d_c, n * n * sizeof(float));

    cudaMemcpy(d_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    blockGemmKernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);

    cudaDeviceSynchronize();

    std::vector<float> c(n * n, 0.0f);

    cudaMemcpy(c.data(), d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
