// Copyright (c) 2024 Soloninko Andrey
#include <iostream>

#include "block_gemm_cuda.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 32

__global__ void BlockGemmCUDAKernel(const float* A, const float* B, float* C,
                                    int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float blockA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float blockB[BLOCK_SIZE][BLOCK_SIZE];

  float sum = 0.0;

  for (int i = 0; i < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
    if (row < n && (i * BLOCK_SIZE + threadIdx.x) < n) {
      blockA[threadIdx.y][threadIdx.x] =
          A[row * n + (i * BLOCK_SIZE + threadIdx.x)];
    } else {
      blockA[threadIdx.y][threadIdx.x] = 0.0;
    }
    if ((i * BLOCK_SIZE + threadIdx.y) < n && col < n) {
      blockB[threadIdx.y][threadIdx.x] =
          B[(i * BLOCK_SIZE + threadIdx.y) * n + col];
    } else {
      blockB[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    for (int j = 0; j < BLOCK_SIZE; ++j) {
      sum += blockA[threadIdx.y][j] * blockB[j][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < n && col < n) {
    C[row * n + col] = sum;
  }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& A,
                                 const std::vector<float>& B, int n) {
  size_t size = n * n * sizeof(float);
  float *d_A, *d_B, *d_C;

  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

  BlockGemmCUDAKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);

  std::vector<float> C(n * n);
  cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return C;
}
