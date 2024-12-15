#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(callable)                                  \
  {                                                                 \
    auto codeError = callable;                                      \
    if (codeError != cudaSuccess) {                                 \
      std::cerr << "CUDA error: " << cudaGetErrorString(codeError)   \
                << '\n';                                             \
      exit(codeError);                                              \
    }                                                               \
  }

__global__ void naive_gemm_kernel(float *c, const float *a, const float *b,
                                  const size_t size) {
  size_t mIdx = blockIdx.y * blockDim.y + threadIdx.y;
  size_t nIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (mIdx < size && nIdx < size) {
    float cVal = 0.0f;
    for (size_t k = 0; k < size; ++k)
        cVal += a[mIdx * size + k] * b[size * k + nIdx];
    c[mIdx * size + nIdx] = cVal;
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
    
    const size_t sizeAxis = 32u;
    dim3 threadsPerBlock(
        sizeAxis,
        sizeAxis
    );
    dim3 numBlocks(
        (size + sizeAxis - 1) / sizeAxis,
        (size + sizeAxis - 1) / sizeAxis
    );
    
    naive_gemm_kernel<<<numBlocks, threadsPerBlock>>>(d_c, d_a, d_b, size);

    cudaMemcpy(c.data(), d_c, sizeInBytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return c;
}