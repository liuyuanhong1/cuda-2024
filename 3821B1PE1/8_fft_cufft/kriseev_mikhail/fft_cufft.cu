#include "fft_cufft.h"
#include <cufft.h>

#include <iostream>

__global__ void normalize_kernel(float *data, int size, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    data[i] = data[i] / n;
  }
}

std::vector<float> FffCUFFT(const std::vector<float> &input, int batch) {
  std::vector<float> output(input.size());

  int n = input.size() / (batch * 2);

  cufftHandle plan;
  cufftComplex *data;

  cudaMalloc((void **)&data, n * batch * sizeof(cufftComplex));
  cudaMemcpy(data, input.data(), input.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  cufftPlanMany(&plan, 1, &n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, batch);

  cufftExecC2C(plan, data, data, CUFFT_FORWARD);

  cudaDeviceSynchronize();

  cufftExecC2C(plan, data, data, CUFFT_INVERSE);

  cudaDeviceSynchronize();

  cufftDestroy(plan);

  int blockSize = 64;
  int numBlocks = (output.size() + blockSize - 1) / blockSize;

  normalize_kernel<<<numBlocks, blockSize>>>(reinterpret_cast<float *>(data),
                                             output.size(), n);

  cudaMemcpy(output.data(), data, output.size() * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(data);

  return output;
}
