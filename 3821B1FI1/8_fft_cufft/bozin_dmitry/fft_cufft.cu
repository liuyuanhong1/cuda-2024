// Copyright (c) 2024 Bozin Dmitry
#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cassert>

__constant__ float input_norm;

__global__ void norm_kernel(float* data, const unsigned n) {
  unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    data[i] *= input_norm;
  }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
  const unsigned n = input.size();
  assert(n % (batch + batch) == 0);
  std::vector<float> res(n);
  const unsigned n_batch = n / batch >> 1;
  const unsigned n_bytes = sizeof(cufftComplex) * n_batch * batch;
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device);
  const unsigned block_size = device_prop.maxThreadsPerBlock;
  const unsigned num_blocks = (n + block_size - 1) / block_size;
  const float norm = 1.0f / static_cast<float>(n_batch);
  cudaMemcpyToSymbol(input_norm, &norm, sizeof(norm));
  cufftComplex* data;
  cudaMalloc(&data, n_bytes);
  cudaMemcpy(data, input.data(), n_bytes, cudaMemcpyHostToDevice);
  cufftHandle handle;
  cufftPlan1d(&handle, n_batch, CUFFT_C2C, batch);
  cufftExecC2C(handle, data, data, CUFFT_FORWARD);
  cufftExecC2C(handle, data, data, CUFFT_INVERSE);
  norm_kernel<<<num_blocks, block_size>>>(reinterpret_cast<float*>(data), n);
  cudaMemcpy(res.data(), data, n_bytes, cudaMemcpyDeviceToHost);
  return res;
}