// Copyright (c) 2024 Nogin Denis
#include <iostream>

#include "block_gemm_cuda.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 32

__global__ void BlockGemmCUDAKernel(const float* A, const float* B, float* C,
                                    int size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

  float sum = 0.0;

  for (int i = 0; i < (size + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
    if (row < size && (i * BLOCK_SIZE + threadIdx.x) < size) {
      sharedA[threadIdx.y][threadIdx.x] =
          A[row * size + (i * BLOCK_SIZE + threadIdx.x)];
    } else {
      sharedA[threadIdx.y][threadIdx.x] = 0.0;
    }
    if ((i * BLOCK_SIZE + threadIdx.y) < size && col < size) {
      sharedB[threadIdx.y][threadIdx.x] =
          B[(i * BLOCK_SIZE + threadIdx.y) * size + col];
    } else {
      sharedB[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    for (int j = 0; j < BLOCK_SIZE; ++j) {
      sum += sharedA[threadIdx.y][j] * sharedB[j][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < size && col < size) {
    C[row * size + col] = sum;
  }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& A,
                                 const std::vector<float>& B, int size) {
  size_t memSize = size * size * sizeof(float);

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;

  cudaMalloc(&d_A, memSize);
  cudaMalloc(&d_B, memSize);
  cudaMalloc(&d_C, memSize);

  cudaMemcpy(d_A, A.data(), memSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), memSize, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks((size + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (size + BLOCK_SIZE - 1) / BLOCK_SIZE);

  BlockGemmCUDAKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, size);

  std::vector<float> C(size * size);

  cudaMemcpy(C.data(), d_C, memSize, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return C;
}
