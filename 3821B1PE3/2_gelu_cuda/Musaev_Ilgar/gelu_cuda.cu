// Copyright (c) 2024 Musaev Ilgar
#include "gelu_cuda.h"
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

__global__ void geluKernel(const float* input, float* output, const int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = input[i];
    output[i] = 0.5f * x * (1.0f + tanh((2.0f / sqrt(3.14f)) * (x + 0.044715f * x * x * x)));
  }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  int n = input.size();
  std::vector<float> output(n);

  // Allocate device memory
  float* d_input;
  float* d_output;
  cudaMalloc(&d_input, n * sizeof(float));
  cudaMalloc(&d_output, n * sizeof(float));

  // Copy input data to device
  cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  geluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);

  // Copy output data from device
  cudaMemcpy(output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_output);

  return output;
}

