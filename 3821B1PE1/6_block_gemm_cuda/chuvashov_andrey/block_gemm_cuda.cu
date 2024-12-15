// Copyright (c) Chuvashov Andrey
#include "block_gemm_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void BlockGemmKernel (const float* a,
                                const float* b,
                                float* c,
                                int n) {

    __shared__ float shared_a[SIZE][SIZE];
    __shared__ float shared_b[SIZE][SIZE];

    int column = blockIdx.x * SIZE + threadIdx.x;
    int row = blockIdx.y * SIZE + threadIdx.y;

    float current = 0.0f;

    if (column < n && row < n){
        for (int i = 0; i < n; i += SIZE) {
            if (i + threadIdx.y < n) {
                shared_a[threadIdx.y][threadIdx.x] =
                a[row * n + i + threadIdx.x];
            }

            if (i + threadIdx.x < n) {
                shared_b[threadIdx.y][threadIdx.x] =
                b[(i + threadIdx.y) * n + column];
            }

            __syncthreads();

            for (int j = 0; j < SIZE; ++j){
                current += shared_a[threadIdx.y][j] * shared_b[j][threadIdx.x];
            }

            __syncthreads();
        }
    }

    c[row * n + column] = current;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n){

    std::vector<float> c(n * n, 0.0f);
    size_t bytesSize = n * n * sizeof(float);

    float* data_a;
    float* data_b;
    float* data_c;

    cudaMalloc(&data_a, bytesSize);
    cudaMalloc(&data_b, bytesSize);
    cudaMalloc(&data_c, bytesSize);

    cudaMemcpy(data_a, a.data(), bytesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(data_b, b.data(), bytesSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(SIZE, SIZE);
    dim3 countOfBlocks((n + SIZE - 1) / SIZE, (n + SIZE - 1) / SIZE);

    BlockGemmKernel<<<countOfBlocks, threadsPerBlock>>>(
        data_a, data_b, data_c, n
    );
    cudaMemcpy(c.data(), data_c, bytesSize, cudaMemcpyDeviceToHost);

    cudaFree(data_a);
    cudaFree(data_b);
    cudaFree(data_c);

    return c;
}