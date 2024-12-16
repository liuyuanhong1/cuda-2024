// Copyright (c) 2024 Nogin Denis
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <iostream>

#include "gelu_cuda.h"

__global__ void GeluKernel(const float* input, float* output, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    float x = input[index];
    float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    output[index] = x * cdf;
  }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  if (input.empty()) return {};
  
  size_t size = input.size();
  std::vector<float> output(size);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  
  auto threadsPerBlock = deviceProp.maxThreadsPerBlock;
  auto numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

  float* inpDev = nullptr;
  float* outDev = nullptr;

  size_t bufferSize = size * sizeof(float);

  cudaMalloc(&inpDev, bufferSize);
  cudaMalloc(&outDev, bufferSize);
  cudaMemcpy(inpDev, input.data(), bufferSize, cudaMemcpyHostToDevice);

  GeluKernel<<<numBlocks, threadsPerBlock>>>(inpDev, outDev, size);

  cudaMemcpy(output.data(), outDev, bufferSize, cudaMemcpyDeviceToHost);

  cudaFree(inpDev);
  cudaFree(outDev);

  return output;
}