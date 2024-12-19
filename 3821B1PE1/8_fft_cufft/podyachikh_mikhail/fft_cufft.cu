// Copyright (c) 2024 Podyachikh Mikhail
#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

__global__ void normalizeKernel(float *input, const int sz, const int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < sz) {
    input[i] /= n;
  }
}

std::vector<float> FffCUFFT(const std::vector<float> &input, int batch) {
  const int sz = input.size();
  std::vector<float> output(sz);
  int n = sz / (batch * 2);

  int sizeInBytes = sizeof(cufftComplex) * n * batch;
  cufftHandle plan;
  cufftPlan1d(&plan, n, CUFFT_C2C, batch);
  cufftComplex *data;

  cudaMalloc(&data, sizeInBytes);
  cudaMemcpy(data, input.data(), sizeInBytes, cudaMemcpyHostToDevice);
  cufftExecC2C(plan, data, data, CUFFT_FORWARD);
  cufftExecC2C(plan, data, data, CUFFT_INVERSE);

  int device;
  cudaGetDevice(&device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  int threadsPerBlock = deviceProp.maxThreadsPerBlock;
  int blockNum = (input.size() + threadsPerBlock - 1) / threadsPerBlock;
  normalizeKernel<<<blockNum, threadsPerBlock>>>(
      reinterpret_cast<float *>(data), sz, n);

  cudaMemcpy(output.data(), data, sizeInBytes, cudaMemcpyDeviceToHost);
  cufftDestroy(plan);
  cudaFree(data);
  return output;
}
