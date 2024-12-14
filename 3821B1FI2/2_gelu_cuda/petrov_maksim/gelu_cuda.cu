#include "gelu_cuda.h"
#include <cuda_runtime.h>


// 2 * sqrt(2.0f / 3.1415926535f) = 1.59577f
__constant__ float coef1 = 1.59577f;
__constant__ float coef2 = 0.044715f;

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float tanh_arg = coef1 * (x + coef2 * x * x * x);
        float exp1 = __expf(tanh_arg);

        // tanh(x) = (e^2x - 1)/(e^2x + 1)
        float tanh_approx = (exp1 - 1) / (exp1 + 1);
        output[idx] = 0.5f * x * (1.0f + tanh_approx);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int size = input.size();

    float* d_input = nullptr;
    float* d_output = nullptr;
    std::vector<float> output(size);

    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    gelu_kernel <<<gridSize, blockSize >>> (d_input, d_output, size);

    cudaMemcpy(output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
