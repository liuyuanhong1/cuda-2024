// Copyright (c) 2024 Musaev Ilgar
#include "gelu_cuda.h"
#include <vector>
#include <cuda_runtime.h>
#include <iostream>

__global__ void geluKernel(const float* input, float* output, size_t s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < s) {
    float x = input[i];
    float x_three = x * x * x;
    float tanh_arg = sqrtf(2.0f / M_PI) * (x + 0.044715f * x_three);
    output[i] = 0.5f * x * (1.0f + tanhf(tanh_arg));
  }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  size_t n = input.size();
  std::vector<float> output(n);
  float* d_input = nullptr;
  float* d_output = nullptr;
  cudaMalloc(&d_input, n * sizeof(float));
  cudaMalloc(&d_output, n * sizeof(float));
  cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  geluKernel<<<numBlocks, blockSize>>>(d_input, d_output, n);
  cudaMemcpy(output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_output);

  return output;
}
