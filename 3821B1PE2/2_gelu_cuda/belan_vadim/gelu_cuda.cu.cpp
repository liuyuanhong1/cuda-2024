#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

__global__ void geluKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); // sqrt(2/pi) ≈ 0.79788
        output[idx] = x * cdf;
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int size = input.size();

    float* h_input = nullptr;
    float* h_output = nullptr;

    h_input = new float[size];
    h_output = new float[size];

    std::copy(input.begin(), input.end(), h_input);

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    geluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);

    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> result(h_output, h_output + size);

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    return result;
}
