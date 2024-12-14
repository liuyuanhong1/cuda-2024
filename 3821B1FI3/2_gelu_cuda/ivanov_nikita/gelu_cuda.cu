#include "gelu_cuda.h"
#include <cmath>
#include <cuda_runtime.h>

// Ядро CUDA для вычисления GELU
__global__ void GeluKernel(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t v_size = input.size();
    std::vector<float> output(v_size);

    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, v_size * sizeof(float));
    cudaMalloc(&d_output, v_size * sizeof(float));

    cudaMemcpy(d_input, input.data(), v_size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (v_size + blockSize - 1) / blockSize;

    GeluKernel<<<numBlocks, blockSize>>>(d_input, d_output, v_size);

    cudaMemcpy(output.data(), d_output, v_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}