#include <cmath>
#include <cstddef>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "naive_gemm_cuda.h"

#define CUDA_CALL(status)                                     \
  {                                                           \
    if (status != cudaSuccess) {                              \
      std::cout << "error " << __FILE__ << ":" << __LINE__;   \
      std::cout << " " << cudaGetErrorString(status) << "\n"; \
      std::exit(0);                                           \
    }                                                         \
  }

__global__ void naive_gemm_kernel(const float* a, const float* b, float* result,
                                  size_t n) {
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  size_t i = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    float res = 0.0f;
    for (int k = 0; k < n; ++k) {
      res += a[i * n + k] * b[k * n + j];
    }
    result[i * n + j] = res;
  }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, int n) {
  size_t size = n * n;
  size_t byte_count = size * sizeof(float);

  float* dev_a_mem = nullptr;
  float* dev_b_mem = nullptr;

  std::vector<float> result(size);
  float* dev_result_mem = nullptr;

  constexpr size_t block_dim_value = 32; // max 1024 threads per block 
  dim3 block_dim(block_dim_value, block_dim_value);
  dim3 grid_dim(ceil(n / block_dim.x), ceil(n / block_dim.y));

  CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_a_mem), byte_count));
  CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_b_mem), byte_count));
  CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_result_mem), byte_count));

  CUDA_CALL(
      cudaMemcpy(dev_a_mem, a.data(), byte_count, cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(dev_b_mem, b.data(), byte_count, cudaMemcpyHostToDevice));

  naive_gemm_kernel<<<grid_dim, block_dim>>>(dev_a_mem, dev_b_mem, dev_result_mem, n);
  CUDA_CALL(cudaGetLastError());

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(result.data()),
                       reinterpret_cast<void*>(dev_result_mem), byte_count,
                       cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(reinterpret_cast<void*>(dev_a_mem)));
  CUDA_CALL(cudaFree(reinterpret_cast<void*>(dev_b_mem)));
  CUDA_CALL(cudaFree(reinterpret_cast<void*>(dev_result_mem)));

  return result;
}
