// Copyright (c) 2024 Morgachev Stepan
#include "gelu_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__global__ void GeluKernel(const float* input, float* output, size_t size) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= size) {
    return;
  }

  const float sqrtTwoOverPi = std::sqrt(2.0f / M_PI);
  constexpr float coeff = 0.044715f;

  float value = input[index];
  float cubicValue = value * value * value;
  float tanhInput = sqrtTwoOverPi * (value + coeff * cubicValue);
  output[index] = 0.5f * value * (1.0f + std::tanh(tanhInput));
}

std::vector<float> GeluCUDA(const std::vector<float>& input){
  size_t size = input.size();

  if (size == 0) {
    return {};
  }

  std::vector<float> output(size);
  float* deviceInput = nullptr;
  float* deviceOutput = nullptr;
  size_t sizeInBytes = size * sizeof(float);

  cudaMalloc(&deviceInput, sizeInBytes);
  cudaMalloc(&deviceOutput, sizeInBytes);

  cudaMemcpy(deviceInput, input.data(), sizeInBytes, cudaMemcpyHostToDevice);

  cudaDeviceProp cudaDeviceProp;
  cudaGetDeviceProperties(&cudaDeviceProp, 0);
  size_t threadsPerBlock = cudaDeviceProp.maxThreadsPerBlock;
  size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  GeluKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceInput, deviceOutput, size);
  cudaMemcpy(output.data(), deviceOutput, sizeInBytes, cudaMemcpyDeviceToHost);

  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  return output;
}
