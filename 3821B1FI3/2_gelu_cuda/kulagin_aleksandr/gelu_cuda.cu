// Copyright (c) 2024 Kulagin Aleksandr
#include "gelu_cuda.h"

#define _USE_MATH_DEFINES
#include <cuda_runtime.h>
#include <cmath>

__constant__ float precalc_c_1;

static const float host_precalc_c_1 = std::sqrt(2.0f / M_PIf);

__global__ void GeluCuda_dev(float* input_output, const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float& x = input_output[i];
    input_output[i] = 0.5f * x * (1.0f + tanhf(precalc_c_1 * ( x + 0.044715f * (x * x * x) )));
  }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  if (input.empty()) {
    return std::vector<float>();
  }
  cudaMemcpyToSymbol(precalc_c_1, &host_precalc_c_1, sizeof(host_precalc_c_1));
  const std::vector<float>::size_type sz = input.size();
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  const int block_size = deviceProp.maxThreadsPerBlock;
  const int num_blocks = (sz + block_size - 1) / block_size;
  std::vector<float> res(sz);
  float* input_output;
  cudaMalloc((void**)&input_output, sz * sizeof(float));
  cudaMemcpy(input_output, input.data(), sz * sizeof(float), cudaMemcpyHostToDevice);
  GeluCuda_dev<<<num_blocks, block_size>>>(input_output, static_cast<int>(sz));
  cudaMemcpy(res.data(), input_output, sz * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(input_output);
  return res;
}
