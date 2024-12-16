// Copyright (c) 2024 Kostanyan Arsen
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <iostream>

#include "gelu_cuda.h"

__global__ void GeluKernel(const float* input, float* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float x = input[idx];
    float cdf =
        0.5f *
        (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    output[idx] = x * cdf;
  }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  if (input.empty()) return {};

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  size_t size = input.size();
  size_t countBytes = size * sizeof(float);
  
  std::vector<float> output(size);
  
  auto threadsPerBlock = deviceProp.maxThreadsPerBlock;
  auto numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

  float* inpDev = nullptr;
  float* outDev = nullptr;

  cudaMalloc(&inpDev, countBytes);
  cudaMalloc(&outDev, countBytes);

  cudaMemcpy(inpDev, input.data(), countBytes, cudaMemcpyHostToDevice);

  GeluKernel<<<numBlocks, threadsPerBlock>>>(inpDev, outDev, size);

  cudaMemcpy(output.data(), outDev, countBytes, cudaMemcpyDeviceToHost);

  cudaFree(inpDev);
  cudaFree(outDev);

  return output;
}
