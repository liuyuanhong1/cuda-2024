// Copyright (c) 2024 Zakharov Artem
#include "block_gemm_cuda.h"
#include "cuda.h"
#include "cuda_runtime.h"

constexpr int BLOCK_SIZE = 32;

__global__ void BlockGemmKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float a_shared[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE * BLOCK_SIZE];

    size_t row = threadIdx.y;
    size_t col = threadIdx.x;
    size_t global_row = blockIdx.y * BLOCK_SIZE + row;
    size_t global_col = blockIdx.x * BLOCK_SIZE + col;

    float sum = 0.0f;

    int num_blocks = gridDim.x;

    for (int i = 0; i < num_blocks; i++) {
        a_shared[row * BLOCK_SIZE + col] = a[global_row * n + (i * BLOCK_SIZE + col)];
        b_shared[row * BLOCK_SIZE + col] = b[(i * BLOCK_SIZE + row) * n + global_col];
        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++) {
            sum += a_shared[row * BLOCK_SIZE + j] * b_shared[j * BLOCK_SIZE + col];
        }
        __syncthreads();
    }

    if (global_row < n && global_col < n) {
        c[global_row * n + global_col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    int size = n * n;
    std::vector<float> c(size);
    size_t bytes_size = size * sizeof(float);
    constexpr size_t threads_per_axis = BLOCK_SIZE;
    const size_t num_blocks_on_axis = (n + threads_per_axis - 1) / threads_per_axis;

    float *a_dev, *b_dev, *c_dev;

    cudaMalloc(reinterpret_cast<void**>(&a_dev), bytes_size);
    cudaMalloc(reinterpret_cast<void**>(&b_dev), bytes_size);
    cudaMalloc(reinterpret_cast<void**>(&c_dev), bytes_size);

    cudaMemcpy(reinterpret_cast<void*>(a_dev),
               reinterpret_cast<const void*>(a.data()),
               bytes_size, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(b_dev),
               reinterpret_cast<const void*>(b.data()),
               bytes_size, cudaMemcpyHostToDevice);

    dim3 num_blocks(num_blocks_on_axis, num_blocks_on_axis);
    dim3 threads_per_block(threads_per_axis, threads_per_axis);

    BlockGemmKernel<<<num_blocks, threads_per_block>>>(a_dev, b_dev, c_dev, n);

    cudaMemcpy(reinterpret_cast<void*>(c.data()),
               reinterpret_cast<const void*>(c_dev),
               bytes_size, cudaMemcpyDeviceToHost);

    cudaFree(reinterpret_cast<void*>(a_dev));
    cudaFree(reinterpret_cast<void*>(b_dev));
    cudaFree(reinterpret_cast<void*>(c_dev));

    return c;
}
