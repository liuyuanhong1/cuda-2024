// Copyright (c) 2024 Gontsov Roman

#include <cstdlib>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gelu_cuda.h"


__global__ void GeluKernel(const float* input, float* output, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= size) return;

  constexpr float geluCoeff1 = 1.595769122f;
  constexpr float geluCoeff2 = 0.071354816f;

  float value = input[i];
  output[i] = value * (1 - 1 / (1.0f + __expf(value * (geluCoeff1 + value * value * geluCoeff2))));
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  if (input.empty()) return {};

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  auto size = input.size();
  std::vector<float> output(size);

  auto sizeInBytes = size * sizeof(float);
  auto threadsPerBlock = deviceProp.maxThreadsPerBlock;
  auto numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

  float *inputDev = nullptr;
  cudaMalloc(&inputDev, sizeInBytes);
  
  float *outputDev = nullptr;
  cudaMalloc(&outputDev, sizeInBytes);

  cudaMemcpy(inputDev, input.data(), sizeInBytes, cudaMemcpyHostToDevice);

  GeluKernel<<<numBlocks, threadsPerBlock>>>(inputDev, outputDev, size);

  cudaDeviceSynchronize();
  cudaMemcpy(output.data(), outputDev, sizeInBytes, cudaMemcpyDeviceToHost);

  cudaFree(outputDev);
  cudaFree(inputDev);
  return output;
}
