#include "gelu_cuda.h"

#define THREADS_PER_BLOCK 256

__global__ void gelu_kernel(const float* input, float* output, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float x = input[i];
        float hyperbolicTan = tanhf(0.7978845608f * (x + 0.044715f * x * x * x));
        output[i] = 0.5 * x * (1 + hyperbolicTan);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t size = input.size();
    size_t bytes = size * sizeof(float);
    float* d_input = nullptr;
    float* d_output = nullptr;
    std::vector<float> output(size);

    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_output, bytes);
    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int threads = THREADS_PER_BLOCK;

    gelu_kernel<<<blocks, threads>>>(d_input, d_output, size);
    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}