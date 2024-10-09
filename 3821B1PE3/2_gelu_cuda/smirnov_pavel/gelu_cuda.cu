#include "gelu_cuda.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void geluKernel(const float* input, float* output, size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    float x = input[i];
    float x3 = x * x * x;
    float arg = 2.0f / 3.14159265359f * (x + 0.044715f * x3);
    output[i] = 0.5f * x * (1.0f + tanh(arg));
  }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  size_t size = input.size();
  std::vector<float> output(size);

  float* d_input;
  float* d_output;
  cudaMalloc(&d_input, size * sizeof(float));
  cudaMalloc(&d_output, size * sizeof(float));

  cudaMemcpy(d_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  geluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);

  cudaMemcpy(output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);

  return output;
}

