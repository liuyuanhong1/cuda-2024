// Copyright (c) 2024 Zakharov Artem
#include "naive_gemm_cuda.h"

__global__ void naive_gemm_kernel(const float* a, const float* b, float* c, int shift) {
    int n = 1 << shift;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n) {
        float cur_res = 0;
        for (size_t i = 0; i < n; i++) {
            cur_res += a[(y << shift) + i] * b[(i << shift) + x];
        }
        c[(y << shift) + x] = cur_res;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    constexpr size_t threads_per_axis = 32;
    const size_t num_blocks_on_axis = (n + threads_per_axis - 1) / threads_per_axis;
    size_t bytes_size = n * n * sizeof(float);
    std::vector<float> c (n * n);

    int shift = 0;
    int tmp = n;
    while (tmp != 1) {
        tmp >>= 1;
        shift++;
    }

    float *a_dev = nullptr;
    float *b_dev = nullptr;
    float *c_dev = nullptr;

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
    naive_gemm_kernel<<<num_blocks, threads_per_block>>>(a_dev, b_dev, c_dev, shift);
    cudaMemcpy(reinterpret_cast<void*>(c.data()),
               reinterpret_cast<const void*>(c_dev),
               bytes_size, cudaMemcpyDeviceToHost);

    cudaFree(reinterpret_cast<void*>(a_dev));
    cudaFree(reinterpret_cast<void*>(b_dev));
    cudaFree(reinterpret_cast<void*>(c_dev));
    return c;
}
