#include <cstddef>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gelu_cuda.h"

#define CUDA_CALL(status)                                     \
  {                                                           \
    if (status != cudaSuccess) {                              \
      std::cout << "error " << __FILE__ << ":" << __LINE__;   \
      std::cout << " " << cudaGetErrorString(status) << "\n"; \
      std::exit(0);                                           \
    }                                                         \
  }

__global__ void gelu_kernel(const float* sample, float* result,
                            size_t elemCount) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < elemCount) {
    float sqrtdvadelpi = 0.797884f;
    float num = sample[id];
    result[id] =
        0.5f * num *
        (1.0f + tanhf(sqrtdvadelpi * num * (1.0f + 0.044715f * num * num)));
  }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  cudaDeviceProp device_prop;
  CUDA_CALL(cudaGetDeviceProperties(&device_prop, 0));

  size_t size = input.size();
  size_t byte_count = size * sizeof(float);

  float* dev_sample_mem = nullptr;

  std::vector<float> result(size);
  float* dev_result_mem = nullptr;

  int block_size = device_prop.maxThreadsPerBlock;
  int num_blocks = (size + block_size - 1) / block_size;

  CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_sample_mem), byte_count));

  CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_result_mem), byte_count));

  CUDA_CALL(cudaMemcpy(dev_sample_mem, input.data(), byte_count,
                       cudaMemcpyHostToDevice));

  gelu_kernel<<<num_blocks, block_size>>>(dev_sample_mem, dev_result_mem, size);
  CUDA_CALL(cudaGetLastError());

  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(result.data()),
                       reinterpret_cast<void*>(dev_result_mem), byte_count,
                       cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(reinterpret_cast<void*>(dev_sample_mem)));
  CUDA_CALL(cudaFree(reinterpret_cast<void*>(dev_result_mem)));

  return result;
}
