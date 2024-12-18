// Copyright (c) 2024 Podyachikh Mikhail
#include "gelu_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void geluCUDA_kernel(float* a, const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float scale = 0.7978845608028653f; // sqrt(2/pi)
        float val = a[idx];
        a[idx] = 0.5f * val * (1.0f + tanhf(scale * val * (1.0f + 0.044715f * val * val)));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    int threadsPerBlock = deviceProp.maxThreadsPerBlock;
    int numBlocks = (input.size() + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<float> output(input);
    float* deviceBuffer = nullptr;

    cudaMalloc(&deviceBuffer, input.size() * sizeof(float));
    cudaMemcpy(deviceBuffer, output.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

    geluCUDA_kernel<<<numBlocks, threadsPerBlock>>>(deviceBuffer, input.size());

    cudaMemcpy(output.data(), deviceBuffer, input.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(deviceBuffer);

    return output;
}
