// Copyright (c) 2024 Kokin Ivan
#include "gelu_cuda.h"
#include <vector>
#include <cuda_runtime.h>
#include <iostream>

__global__ void geluKernel(const float* input, float* output, size_t s) {
  int src = blockIdx.x * blockDim.x + threadIdx.x;
  if (src < s) {
    float x = input[src];
    float tanh_arg = sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x);
    output[src] = 0.5f * x * (1.0f + tanhf(tanh_arg));
  }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  size_t net = input.size();
  std::vector<float> output(net);
  float* n_input = nullptr;
  float* n_output = nullptr;
  cudaMalloc(&n_input, net * sizeof(float));
  cudaMalloc(&n_output, net * sizeof(float));
  cudaMemcpy(n_input, input.data(), net * sizeof(float), cudaMemcpyHostToDevice);
  int blockSize = 256;
  int numBlocks = (net + blockSize - 1) / blockSize;
  geluKernel<<<numBlocks, blockSize>>>(n_input, n_output, net);
  cudaMemcpy(output.data(), n_output, net * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(n_input);
  cudaFree(n_output);

  return output;
}
