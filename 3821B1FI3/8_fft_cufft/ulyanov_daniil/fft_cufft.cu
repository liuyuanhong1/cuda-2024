// Copyright (c) 2024 Ulyanov Daniil

#include <cstdlib>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "device_launch_parameters.h"
#include "fft_cufft.h"

#define VERIFY_CUDA_CALL(call)                                                 \
  {                                                                            \
    auto errorCode = call;                                                     \
    if (errorCode != cudaSuccess) {                                            \
      std::cerr << "\033[1;31mCUDA Error:\033[0m ";                            \
      std::cerr << cudaGetErrorString(errorCode) << '\n';                      \
      std::cerr << "Error Code: " << static_cast<int>(errorCode) << '\n';      \
      std::cerr << "Location: " << __FILE__ << " (" << __LINE__ << ")\n";      \
      std::exit(static_cast<int>(errorCode));                                  \
    }                                                                          \
  }

#define VERIFY_CUFFT_CALL(call)                                                \
  {                                                                            \
    auto cufftStatus = call;                                                   \
    if (cufftStatus != CUFFT_SUCCESS) {                                        \
      std::cerr << "\033[1;31mcuFFT Error:\033[0m ";                           \
      std::cerr << static_cast<int>(cufftStatus) << '\n';                      \
      std::cerr << "Location: " << __FILE__ << " (" << __LINE__ << ")\n";      \
      std::exit(static_cast<int>(cufftStatus));                                \
    }                                                                          \
  }

__global__ void apply_normalization(float *data, size_t length, float factor) {
  size_t idx =
      static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) +
      static_cast<size_t>(threadIdx.x);
  if (idx < length) {
    data[idx] *= factor;
  }
}

std::vector<float> FffCUFFT(const std::vector<float> &input, int batch) {
  cudaDeviceProp deviceProperties{};
  VERIFY_CUDA_CALL(cudaGetDeviceProperties(&deviceProperties, 0));

  if (input.empty())
    return {};

  auto totalSize = input.size();
  auto itemsPerBatch = totalSize / batch >> 1;
  auto byteCount = sizeof(cufftComplex) * itemsPerBatch * batch;
  auto maxThreads = deviceProperties.maxThreadsPerBlock;
  auto totalBlocks = (totalSize + maxThreads - 1) / maxThreads;
  std::vector<float> result(totalSize);

  cufftComplex *deviceSignal = nullptr;
  VERIFY_CUDA_CALL(cudaMalloc(&deviceSignal, byteCount));
  VERIFY_CUDA_CALL(cudaMemcpy(reinterpret_cast<void *>(deviceSignal),
                              reinterpret_cast<const void *>(input.data()),
                              byteCount, cudaMemcpyHostToDevice));

  cufftHandle fftHandle{};
  VERIFY_CUFFT_CALL(cufftPlan1d(&fftHandle, itemsPerBatch, CUFFT_C2C, batch));
  VERIFY_CUFFT_CALL(
      cufftExecC2C(fftHandle, deviceSignal, deviceSignal, CUFFT_FORWARD));
  VERIFY_CUFFT_CALL(
      cufftExecC2C(fftHandle, deviceSignal, deviceSignal, CUFFT_INVERSE));
  VERIFY_CUFFT_CALL(cufftDestroy(fftHandle));

  apply_normalization<<<totalBlocks, maxThreads>>>(
      reinterpret_cast<float *>(deviceSignal), totalSize, 1.0f / itemsPerBatch);
  VERIFY_CUDA_CALL(cudaDeviceSynchronize());
  VERIFY_CUDA_CALL(cudaGetLastError());

  VERIFY_CUDA_CALL(cudaMemcpy(reinterpret_cast<void *>(result.data()),
                              reinterpret_cast<void *>(deviceSignal), byteCount,
                              cudaMemcpyDeviceToHost));
  VERIFY_CUDA_CALL(cudaFree(reinterpret_cast<void *>(deviceSignal)));

  return result;
}
