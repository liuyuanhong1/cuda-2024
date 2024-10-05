// Copyright (c) 2024 Kulikov Artem
#include <vector>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda.h>

__global__ void myKernel(const float *a, const float *b,
                            float *const c, const size_t size) {
  size_t mIdx = blockIdx.y * blockDim.y + threadIdx.y;
  size_t kIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (mIdx < size && kIdx < size) {
    float *const c_ = &c[mIdx * size]; 
    float const cnst = a[mIdx * size + kIdx];
    const float *const b_ = &b[size * kIdx];

    for (int n = 0; n < size; n+=2) {
        c_[n] += cnst * b_[n];
        c_[n+1] += cnst * b_[n+1];
    }
  }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
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
    
    const size_t sizeAxis = 64u;
    dim3 threadsPerBlock(
        sizeAxis,
        sizeAxis
    );
    dim3 numBlocks(
        (size + sizeAxis - 1) / sizeAxis,
        (size + sizeAxis - 1) / sizeAxis
    );
    
    myKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, size);

    cudaMemcpy(c.data(), d_c, sizeInBytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return c;
}
