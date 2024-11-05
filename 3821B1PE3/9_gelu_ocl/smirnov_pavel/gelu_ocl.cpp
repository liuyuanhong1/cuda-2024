#include <cmath>
#include <cstddef>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"

#include "fft_cufft.h"

#define CUDA_CALL(status)                                     \
  {                                                           \
    if (status != cudaSuccess) {                              \
      std::cout << "error " << __FILE__ << ":" << __LINE__;   \
      std::cout << " " << cudaGetErrorString(status) << "\n"; \
      std::exit(0);                                           \
    }                                                         \
  }

#define CUFFT_CALL(status)                                          \
  {                                                                 \
    if (status != CUFFT_SUCCESS) {                                  \
      std::cout << "error " << __FILE__ << ":" << __LINE__ << "\n"; \
      std::exit(0);                                                 \
    }                                                               \
  }

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
  size_t size = input.size();
  size_t byte_count = size * sizeof(float);
  size_t batch_size = size / batch / 2;

  cufftComplex* dev_input_mem = nullptr;
  std::vector<float> result(size);

  CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_input_mem), byte_count));

  CUDA_CALL(cudaMemcpy(dev_input_mem, input.data(), byte_count,
                       cudaMemcpyHostToDevice));

  cufftHandle plan;

  CUFFT_CALL(cufftPlan1d(&plan, batch_size, CUFFT_C2C, batch));
  CUFFT_CALL(cufftExecC2C(plan, dev_input_mem, dev_input_mem, CUFFT_FORWARD));
  CUFFT_CALL(cufftExecC2C(plan, dev_input_mem, dev_input_mem, CUFFT_INVERSE));

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(result.data()),
                       reinterpret_cast<void*>(dev_input_mem), byte_count,
                       cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < size; i++) {
    result[i] /= static_cast<float>(batch_size);
  }

  cufftDestroy(plan);
  CUDA_CALL(cudaFree(reinterpret_cast<void*>(dev_input_mem)));

  return result;
}