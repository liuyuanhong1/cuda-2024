#include <cmath>
#include <cstddef>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "block_gemm_cuda.h"

#define CUDA_CALL(status)                                     \
  {                                                           \
    if (status != cudaSuccess) {                              \
      std::cout << "error " << __FILE__ << ":" << __LINE__;   \
      std::cout << " " << cudaGetErrorString(status) << "\n"; \
      std::exit(0);                                           \
    }                                                         \
  }

constexpr size_t block_dim_value = 32;  // max 1024 threads per block

__global__ void naive_gemm_kernel(const float* a, const float* b, float* result,
                                  size_t n) {
  int local_x = threadIdx.x;
  int local_y = threadIdx.y;
  int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  int global_y = blockIdx.y * blockDim.y + threadIdx.y;

  int grid_dim = gridDim.x;

  __shared__ float a_block[block_dim_value][block_dim_value];
  __shared__ float b_block[block_dim_value][block_dim_value];

  float c_value = 0.0f;

  int a_idx = global_y * n + local_x;
  int b_idx = local_y * n + global_x;

  for (int block_k = 0; block_k < grid_dim; ++block_k) {
    a_block[local_y][local_x] =
        a[a_idx + block_k * block_dim_value];
    b_block[local_y][local_x] =
        b[b_idx + block_k * block_dim_value * n];
    __syncthreads();  
    
    for (int k = 0; k < block_dim_value; k++) {
      c_value += a_block[local_y][k] * b_block[k][local_x];
    }
  }
  __syncthreads();     

  result[global_y * n + global_x] = c_value;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, int n) {
  size_t size = n * n;
  size_t byte_count = size * sizeof(float);

  float* dev_a_mem = nullptr;
  float* dev_b_mem = nullptr;

  std::vector<float> result(size);
  float* dev_result_mem = nullptr;

  dim3 block_dim(block_dim_value, block_dim_value);
  dim3 grid_dim(ceil(n / block_dim.x), ceil(n / block_dim.y));

  CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_a_mem), byte_count));
  CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_b_mem), byte_count));
  CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_result_mem), byte_count));

  CUDA_CALL(
      cudaMemcpy(dev_a_mem, a.data(), byte_count, cudaMemcpyHostToDevice));
  CUDA_CALL(
      cudaMemcpy(dev_b_mem, b.data(), byte_count, cudaMemcpyHostToDevice));

  naive_gemm_kernel<<<grid_dim, block_dim>>>(dev_a_mem, dev_b_mem,
                                             dev_result_mem, n);
  CUDA_CALL(cudaGetLastError());

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(result.data()),
                       reinterpret_cast<void*>(dev_result_mem), byte_count,
                       cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(reinterpret_cast<void*>(dev_a_mem)));
  CUDA_CALL(cudaFree(reinterpret_cast<void*>(dev_b_mem)));
  CUDA_CALL(cudaFree(reinterpret_cast<void*>(dev_result_mem)));

  return result;
}
