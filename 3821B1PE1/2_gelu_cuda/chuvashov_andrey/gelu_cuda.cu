// Copyright (c) 2024 Chuvashov Andrey
#include "gelu_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__global__ void GeluKernel(const float* input, float* result, size_t size) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  const float pi_c = sqrt(2.0f / (2 * asin(1.0f)));

  if (index < size) {
    float x = input[index];
    result[index] = 0.5f * x * (1.0f + tanhf(pi_c * (x + PER_C * x * x * x)));
  }
}

std::vector<float> GeluCUDA(const std::vector<float>& input){
  size_t length = input.size();
  std::vector<float> result(length);
  size_t dataLength = length * sizeof(*input.data());

  float* dInput;
  float* dOutput;
  cudaMalloc(&dInput, dataLength);
  cudaMalloc(&dOutput, dataLength);

  cudaMemcpy(dInput, input.data(), dataLength, cudaMemcpyHostToDevice);

  cudaDeviceProp devPropts;
  cudaGetDeviceProperties(&devPropts, 0);
  size_t threadsPerBlock = devPropts.maxThreadsPerBlock;
  size_t countOfBlocks = (length + threadsPerBlock - 1) / threadsPerBlock;

  GeluKernel<<<countOfBlocks, threadsPerBlock>>>(dInput, dOutput, length);
  cudaMemcpy(result.data(), dOutput, dataLength, cudaMemcpyDeviceToHost);

  cudaFree(dInput);
  cudaFree(dOutput);

  return result;
}
