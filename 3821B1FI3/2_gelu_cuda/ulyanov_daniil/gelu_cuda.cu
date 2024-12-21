// Copyright (c) 2024 Ulyanov Daniil

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "gelu_cuda.h"

__global__ void GeluKernel(const float *input, float *output, int size) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadId < size) {
    float value = input[threadId];
    float gelu_cdf =
        0.5f * (1.0f + tanhf(0.7978845608028654f *
                             (value + 0.044715f * value * value * value)));
    output[threadId] = value * gelu_cdf;
  }
}

std::vector<float> GeluCUDA(const std::vector<float> &input) {
  if (input.empty())
    return {};

  cudaDeviceProp gpuProperties;
  cudaGetDeviceProperties(&gpuProperties, 0);

  size_t arraySize = input.size();
  size_t byteSize = arraySize * sizeof(float);

  std::vector<float> hostOutput(arraySize);

  int threadsInBlock = gpuProperties.maxThreadsPerBlock;
  int blocksCount = (arraySize + threadsInBlock - 1) / threadsInBlock;

  float *deviceInput = nullptr;
  float *deviceOutput = nullptr;

  cudaMalloc(&deviceInput, byteSize);
  cudaMalloc(&deviceOutput, byteSize);

  cudaMemcpy(deviceInput, input.data(), byteSize, cudaMemcpyHostToDevice);

  GeluKernel<<<blocksCount, threadsInBlock>>>(deviceInput, deviceOutput,
                                              arraySize);

  cudaMemcpy(hostOutput.data(), deviceOutput, byteSize, cudaMemcpyDeviceToHost);

  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  return hostOutput;
}
