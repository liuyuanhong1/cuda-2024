// Copyright (c) 2024 Kudinov Nikita

#include <cstdlib>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "naive_gemm_cuda.h"

#define BLOCK_SIZE 32

__global__ void NaiveGemmKernel(const float* a, const float* b, float* c,
                                 const size_t size)
{
    constexpr auto blockSize = BLOCK_SIZE;
    __shared__ float sharedA[blockSize][blockSize];
    __shared__ float sharedB[blockSize][blockSize];

    size_t iIdx = blockIdx.y * blockSize + threadIdx.y;
    size_t jIdx = blockIdx.x * blockSize + threadIdx.x;

    float result = 0.0f;

    for (size_t k = 0; k < size; k += blockSize) {

        if (jIdx < size && (threadIdx.y + k)  < size) {
            sharedB[threadIdx.y][threadIdx.x] = __ldg(&b[(threadIdx.y + k) * size + jIdx]);
        } else {
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (iIdx < size && (threadIdx.x + k) < size) {
            sharedA[threadIdx.y][threadIdx.x] = __ldg(&a[iIdx * size + threadIdx.x + k]);
        } else {
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (size_t l = 0; l < blockSize; ++l) {
            result += sharedA[threadIdx.y][l] * sharedB[l][threadIdx.x];
        }

        __syncthreads();
    }

    if (iIdx < size && jIdx < size) {
        c[iIdx * size + jIdx] = result;
    }
}


std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, int size) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  auto countElem = size * size;
  std::vector<float> output(countElem);
  auto sizeInBytes = countElem * sizeof(float);

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

  NaiveGemmKernel<<<numBlocks, threadsPerBlock>>>(aDev, bDev, cDev, size);

  cudaDeviceSynchronize();
  cudaMemcpy(output.data(), cDev, sizeInBytes, cudaMemcpyDeviceToHost);

  cudaFree(cDev);
  cudaFree(bDev);
  cudaFree(aDev);

  return output;
}
