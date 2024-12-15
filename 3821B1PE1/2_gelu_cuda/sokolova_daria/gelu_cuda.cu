// Copyright (c) 2024 Sokolova Daria
#include "gelu_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

global void GeluKernel(const float* input, float* output, size_t size) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= size) {
    return;
  }

  const float factor = std::sqrt(2.0f / M_PI);
  constexpr float cubicCoeff = 0.044715f;

  float curr = input[index];
  output[index] = 0.5f * curr * (1.0f + std::tanh(factor * (curr + cubicCoeff * curr * curr * curr)));
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  size_t size = input.size();

  if (size == 0) {
    return {};
  }

  std::vector<float> output(size);
  float* deviceInputArray = nullptr;
  float* deviceOutputArray = nullptr;
  size_t bufferSize = size * sizeof(float);

  cudaMalloc(&deviceInputArray, bufferSize);
  cudaMalloc(&deviceOutputArray, bufferSize);

  cudaMemcpy(deviceInputArray, input.data(), bufferSize, cudaMemcpyHostToDevice);

  cudaDeviceProp deviceProperties;
  cudaGetDeviceProperties(&deviceProperties, 0);
  size_t threadsPerBlock = deviceProperties.maxThreadsPerBlock;
  size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  GeluKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceInputArray, deviceOutputArray, size);
  cudaMemcpy(output.data(), deviceOutputArray, bufferSize, cudaMemcpyDeviceToHost);

  cudaFree(deviceInputArray);
  cudaFree(deviceOutputArray);

  return output;
}
