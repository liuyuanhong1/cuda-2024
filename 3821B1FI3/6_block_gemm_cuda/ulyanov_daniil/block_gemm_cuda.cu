// Copyright (c) 2024 Ulyanov Daniil

#include <iostream>

#include "block_gemm_cuda.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define TILE_SIZE 32

__global__ void BlockGemmCUDAKernel(const float *A, const float *B, float *C,
                                    int n) {
  int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
  int globalCol = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE];

  float accumulatedSum = 0.0;

  for (int tileIdx = 0; tileIdx < (n + TILE_SIZE - 1) / TILE_SIZE; ++tileIdx) {
    if (globalRow < n && (tileIdx * TILE_SIZE + threadIdx.x) < n) {
      tileA[threadIdx.y][threadIdx.x] =
          A[globalRow * n + (tileIdx * TILE_SIZE + threadIdx.x)];
    } else {
      tileA[threadIdx.y][threadIdx.x] = 0.0;
    }

    if ((tileIdx * TILE_SIZE + threadIdx.y) < n && globalCol < n) {
      tileB[threadIdx.y][threadIdx.x] =
          B[(tileIdx * TILE_SIZE + threadIdx.y) * n + globalCol];
    } else {
      tileB[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    for (int elem = 0; elem < TILE_SIZE; ++elem) {
      accumulatedSum += tileA[threadIdx.y][elem] * tileB[elem][threadIdx.x];
    }

    __syncthreads();
  }

  if (globalRow < n && globalCol < n) {
    C[globalRow * n + globalCol] = accumulatedSum;
  }
}

std::vector<float> BlockGemmCUDA(const std::vector<float> &A,
                                 const std::vector<float> &B, int n) {
  size_t totalSize = n * n * sizeof(float);
  float *deviceA, *deviceB, *deviceC;

  cudaMalloc(&deviceA, totalSize);
  cudaMalloc(&deviceB, totalSize);
  cudaMalloc(&deviceC, totalSize);

  cudaMemcpy(deviceA, A.data(), totalSize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, B.data(), totalSize, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  dim3 blocksPerGrid((n + TILE_SIZE - 1) / TILE_SIZE,
                     (n + TILE_SIZE - 1) / TILE_SIZE);

  BlockGemmCUDAKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB,
                                                          deviceC, n);

  std::vector<float> result(n * n);
  cudaMemcpy(result.data(), deviceC, totalSize, cudaMemcpyDeviceToHost);

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  return result;
}
