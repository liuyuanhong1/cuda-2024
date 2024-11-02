#include "gelu_cuda.h"

__global__ void GeluKernel(float *input, float *res, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float x = input[i];
  res[i] = x * (1.0f -
                1.0f / (1.0f + __expf(x * fma(x * x, GELU_COEF2, GELU_COEF1))));
}

std::vector<float> GeluCUDA(const std::vector<float> &input) {
  auto size = input.size();
  std::vector<float> output(size);
  float *d_input, *d_output;

  cudaMalloc(&d_input, input.size() * sizeof(float));
  cudaMalloc(&d_output, output.size() * sizeof(float));
  cudaMemcpy(d_input, input.data(), size * sizeof(float),
             cudaMemcpyHostToDevice);

  int blockSize = 128;
  int numBlocks = (input.size() + blockSize - 1) / blockSize;

  GeluKernel<<<numBlocks, blockSize>>>(d_input, d_output, size);

  cudaDeviceSynchronize();
  cudaMemcpy(output.data(), d_output, size * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_output);
  cudaFree(d_input);
  return output;
}
