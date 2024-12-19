// Copyright (c) 2024 Filatov Maxim

#include <cstdlib>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "block_gemm_cuda.h"


#define BLOCK_SIZE 32

__global__ void BlockGemmKernel(const float* a, const float* b, float* c, const size_t size) {
  __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE + 1];
  __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE + 1];

  const int threadX = threadIdx.x;
  const int threadY = threadIdx.y;
  const int matrixRow = blockIdx.y * BLOCK_SIZE + threadY;
  const int matrixCol = blockIdx.x * BLOCK_SIZE + threadX;
  const int rowOffset = matrixRow * size;
  const bool isRowInRange = matrixRow < size;
  const bool isColInRange = matrixCol < size;

  float resultValue = 0.0f;

  for (int tileIndex = 0; tileIndex < size / BLOCK_SIZE; ++tileIndex) {
      const int tileStart = tileIndex * BLOCK_SIZE;
      const int aIndex = rowOffset + tileStart + threadX;
      const int bIndex = (tileStart + threadY) * size + matrixCol;

      sharedA[threadY][threadX] = (isRowInRange && (tileStart + threadX) < size) ? __ldg(&a[aIndex]) : 0.0f;
      sharedB[threadY][threadX] = (isColInRange && (tileStart + threadY) < size) ? __ldg(&b[bIndex]) : 0.0f;

      __syncthreads();

      #pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
          resultValue += sharedA[threadY][k] * sharedB[k][threadX];
      }

      __syncthreads();
  }

  if (isRowInRange && isColInRange) {
      c[rowOffset + matrixCol] = resultValue;
  }
}


std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, int size) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  const auto countElem = size * size;
  std::vector<float> output(countElem);
  const auto sizeInBytes = countElem * sizeof(float);

  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  auto t = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 numBlocks(t, t);

  float *aDev = nullptr;
  cudaMalloc(&aDev, sizeInBytes);

  float *bDev = nullptr;
  cudaMalloc(&bDev, sizeInBytes);

  float *cDev = nullptr;
  cudaMalloc(&cDev, sizeInBytes);

  cudaMemcpy(aDev, a.data(), sizeInBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(bDev, b.data(), sizeInBytes, cudaMemcpyHostToDevice);

  BlockGemmKernel<<<numBlocks, threadsPerBlock>>>(aDev, bDev, cDev, size);

  cudaDeviceSynchronize();
  cudaMemcpy(output.data(), cDev, sizeInBytes, cudaMemcpyDeviceToHost);

  cudaFree(cDev);
  cudaFree(bDev);
  cudaFree(aDev);

  return output;
}