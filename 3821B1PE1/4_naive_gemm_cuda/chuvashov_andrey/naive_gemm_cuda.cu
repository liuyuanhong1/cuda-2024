// Copyright (c) 2024 Chuvashov Andrey
#include "naive_gemm_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void NaiveGemmKernel(const float* a,
                                const float* b,
                                float* c,
                                int n) {
  
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (column < n && row < n) {
        float current = 0.0f;
        for (int i = 0; i < n; i++){
            current += a[row * n + i] * b[i * n + column];
        }
        c[row * n + column] = current;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
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

    NaiveGemmKernel<<<countOfBlocks, threadsPerBlock>>>(
        data_a, data_b, data_c, n
    );
    cudaMemcpy(c.data(), data_c, bytesSize, cudaMemcpyDeviceToHost);

    cudaFree(data_a);
    cudaFree(data_b);
    cudaFree(data_c);

    return c;
}
