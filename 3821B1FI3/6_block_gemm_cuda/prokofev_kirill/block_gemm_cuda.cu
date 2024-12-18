// Copyright (c) 2024 Prokofev Kirill
#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cassert>

constexpr unsigned BLOCK_SIZE = 32;

__global__ void BlockGemmCUDA_dev(const float* a, const float* b, float* res, const unsigned n) {
  const unsigned tIx = threadIdx.x;
  const unsigned tIy = threadIdx.y;
  const unsigned i = blockIdx.y * blockDim.y + tIy;
  const unsigned j = blockIdx.x * blockDim.x + tIx;
  __shared__ float shared_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float shared_b[BLOCK_SIZE][BLOCK_SIZE];
  float tmp_sum = 0.0f;
  const unsigned num_blocks_x = gridDim.x;
  for (unsigned k = 0; k < num_blocks_x; k++) {
    shared_a[tIy][tIx] = a[i * n + k * BLOCK_SIZE + tIx];
    shared_b[tIy][tIx] = b[k * BLOCK_SIZE * n + tIy * n + j];
    __syncthreads();
    for (unsigned l = 0; l < BLOCK_SIZE; l++) {
      tmp_sum += shared_a[tIy][l] * shared_b[l][tIx];
    }
    __syncthreads();
  }
  if (i < n && j < n) {
    res[i * n + j] = tmp_sum;
  }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
  const unsigned _n = n;
  const unsigned sz = _n * _n;
  const unsigned sz_bytes = sz * sizeof(float);
  if (n <= 0) {
    return std::vector<float>();
  }
  assert(a.size() == sz);
  assert(b.size() == sz);
  std::vector<float> res(sz);
  const unsigned& block_dim_x = BLOCK_SIZE;
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
  BlockGemmCUDA_dev<<<num_blocks, block_size>>>(a_dev, b_dev, res_dev, _n);
  cudaMemcpy(res.data(), res_dev, sz_bytes, cudaMemcpyDeviceToHost);
  return res;
}
