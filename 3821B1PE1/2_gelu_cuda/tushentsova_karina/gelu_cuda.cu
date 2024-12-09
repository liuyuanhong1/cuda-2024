// Copyright (c) 2024 Tushentsova Karina
#include "gelu_cuda.h"

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void GeluKernel(const float* input, float* output, size_t size) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
      const float sqrtOver = std::sqrt(2.0f / M_PI);
      constexpr float gelu_coeff = 0.044715f;

      float value = input[index];
      float cubValue = value * value * value;
      float tanIn = sqrtOver * (value + gelu_coeff * cubValue);
      output[index] = 0.5f * value * (1.0f + std::tanh(tanIn));
  }
}

std::vector<float> GeluCUDA(const std::vector<float>& input){
  size_t size = input.size();

  if (size == 0) {
    return {};
  }

  std::vector<float> output(size);
  float* deviceIn = nullptr;
  float* deviceOut = nullptr;
  size_t memorySize = size * sizeof(float);

  cudaMalloc(&deviceIn, memorySize);
  cudaMalloc(&deviceOut, memorySize);

  cudaMemcpy(deviceIn, input.data(), memorySize, cudaMemcpyHostToDevice);

  cudaDeviceProp cudaDeviceProp;
  cudaGetDeviceProperties(&cudaDeviceProp, 0);
  size_t threadsPerBlock = cudaDeviceProp.maxThreadsPerBlock;
  size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  GeluKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceIn, deviceOut, size);
  cudaMemcpy(output.data(), deviceOut, memorySize, cudaMemcpyDeviceToHost);

  cudaFree(deviceIn);
  cudaFree(deviceOut);

  return output;
}