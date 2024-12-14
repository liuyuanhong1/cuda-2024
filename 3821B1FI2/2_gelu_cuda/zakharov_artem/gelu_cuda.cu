// Copyright (c) 2024 Zakharov Artem
#include "gelu_cuda.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void gelu_kernel(const float* input, float* output, size_t n) {
    constexpr float SQRT_TWO_OVER_PI = 0.797885;
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < n) {
        float x = input[ind];
        output[ind] = 0.5f * x * (1 + static_cast<float>(tanh(SQRT_TWO_OVER_PI * x * (1 + 0.044715f * x * x))));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);
    size_t n = input.size();
    size_t bytes_size = n * sizeof(float);
    const size_t threads_per_block = dev_prop.maxThreadsPerBlock;
    const size_t num_blocks = (n + threads_per_block - 1) / threads_per_block;
    std::vector<float> result(n);

    float *input_dev = nullptr;
    float *output_dev = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&input_dev), bytes_size);
    cudaMalloc(reinterpret_cast<void**>(&output_dev), bytes_size);
    cudaMemcpy(reinterpret_cast<void*>(input_dev),
               reinterpret_cast<const void*>(input.data()),
               bytes_size, cudaMemcpyHostToDevice);

    gelu_kernel<<<num_blocks, threads_per_block>>>(input_dev, output_dev, n);
    cudaMemcpy(reinterpret_cast<void*>(result.data()),
               reinterpret_cast<const void*>(output_dev),
               bytes_size, cudaMemcpyDeviceToHost);

    cudaFree(reinterpret_cast<void*>(input_dev));
    cudaFree(reinterpret_cast<void*>(output_dev));
    return result;
}
