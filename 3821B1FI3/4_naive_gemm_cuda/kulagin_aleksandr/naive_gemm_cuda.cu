// Copyright (c) 2024 Kulagin Aleksandr
#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cmath>

__global__ void NaiveGemmCUDA_dev(const float* a, const float* b, float* res, const unsigned n) {
  const unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && j < n) {
    float tmp = 0.0f;
    for (unsigned k = 0; k < n; k++) {
      tmp += a[i * n + k] * b[k * n + j];
    }
    res[i * n + j] = tmp;
  }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
  const unsigned _n = n;
  const unsigned sz = _n * _n;
  const unsigned sz_bytes = sz * sizeof(float);
  if ((int)a.size() != sz || (int)b.size() != sz) {
    return std::vector<float>();
  }
  std::vector<float> res(sz);
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device);
  const unsigned block_dim_x = std::sqrt(device_prop.maxThreadsPerBlock);
  const unsigned num_blocks_x = (_n + block_dim_x - 1) / block_dim_x;
  dim3 block_size(block_dim_x, block_dim_x);
  dim3 num_blocks(num_blocks_x, num_blocks_x);
  float* a_dev;
  float* b_dev;
  float* res_dev;
  cudaMalloc((void**)&a_dev, sz_bytes);
  cudaMalloc((void**)&b_dev, sz_bytes);
  cudaMalloc((void**)&res_dev, sz_bytes);
  cudaMemcpy(a_dev, a.data(), sz_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b.data(), sz_bytes, cudaMemcpyHostToDevice);
  NaiveGemmCUDA_dev<<<num_blocks, block_size>>>(a_dev, b_dev, res_dev, _n);
  cudaMemcpy(res.data(), res_dev, sz_bytes, cudaMemcpyDeviceToHost);
  return res;
}
