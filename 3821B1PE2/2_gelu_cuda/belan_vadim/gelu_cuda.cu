#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>

#define CUDA_CHECK_ERROR(call)                                          \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            throw std::runtime_error(std::string("CUDA Error: ") +     \
                                    cudaGetErrorString(err));          \
        }                                                               \
    } while (0)

__global__ void gelu_kernel(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf;
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t size = input.size();
    size_t bytes = size * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_input, bytes));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_output, bytes));

    CUDA_CHECK_ERROR(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    gelu_kernel<<<blocks, threads>>>(d_input, d_output, size);

    CUDA_CHECK_ERROR(cudaGetLastError());

    std::vector<float> output(size);

    CUDA_CHECK_ERROR(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaFree(d_input));
    CUDA_CHECK_ERROR(cudaFree(d_output));

    return output;
}
