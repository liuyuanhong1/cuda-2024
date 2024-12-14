#include "gelu_cuda.h"
#include <cmath>
#include <cuda_runtime.h>

__global__ void GeluKernel(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t inputCount = input.size();
    std::vector<float> output(inputCount);

    float* deviceInput;
    float* deviceOutput;
    cudaMalloc(&deviceInput, inputCount * sizeof(float));
    cudaMalloc(&deviceOutput, inputCount * sizeof(float));

    cudaMemcpy(deviceInput, input.data(), inputCount * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (inputCount + blockSize - 1) / blockSize;

    GeluKernel<<<numBlocks, blockSize>>>(deviceInput, deviceOutput, inputCount);

    cudaMemcpy(output.data(), deviceOutput, inputCount * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    return output;
}