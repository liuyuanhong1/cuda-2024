#include <vector>
#include <cstdlib>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

const float sqrt2pi = 0.797884f;

__global__ void kernel(const float* sample, float* result,
                            size_t elemCount) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < elemCount) {
    const float num = sample[id];
    result[id] = 0.5f * num *
                 (1.0f + tanhf(sqrt2pi * num * (1.0f + 0.044715f * num * num)));
  }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  const size_t size = input.size();
  std::vector<float> output(size);

  size_t sizeInBytes = size * sizeof(*input.data());

  float* d_input;
  float* d_output;
  cudaMalloc(&d_input, sizeInBytes);
  cudaMalloc(&d_output, sizeInBytes);

  cudaMemcpy(d_input, input.data(), sizeInBytes, cudaMemcpyHostToDevice);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  size_t threadsPerBlock = deviceProp.maxThreadsPerBlock;
  size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);

  cudaMemcpy(output.data(), d_output, sizeInBytes, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
  return output;
}
