#include "fft_cufft.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

__global__ void norm_kernel(float* a, const int n, const int nx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    a[i] /= nx;
  }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  int threadsPerBlock = deviceProp.maxThreadsPerBlock;
  int blockNum = (input.size() + threadsPerBlock - 1) / threadsPerBlock;

  size_t nx = input.size() / batch / 2;

  cufftComplex* ptr;
  cudaMalloc(&ptr, sizeof(float) * input.size());
  cudaMemcpy(ptr, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice);

  cufftHandle plan;
  cufftPlan1d(&plan, nx, CUFFT_C2C, batch);
  cufftExecC2C(plan, ptr, ptr, CUFFT_FORWARD);
  cufftExecC2C(plan, ptr, ptr, CUFFT_INVERSE);

  std::vector<float> result(input.size());
  norm_kernel<<<blockNum, threadsPerBlock>>>(reinterpret_cast<float*>(ptr), input.size(), nx);
  cudaMemcpy(result.data(), ptr, sizeof(float) * input.size(), cudaMemcpyDeviceToHost);

  cufftDestroy(plan);
  cudaFree(ptr);

  return result;
}
