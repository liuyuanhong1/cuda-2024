#include <cmath>

#include "gelu_cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void geluKernel(const float* input, float* output, size_t size_input) 
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr float temp = 0.7978845608f;

    if (i < size_input) 
    {
        float x = input[i];
        output[i] = 0.5f * x * (1.f + tanhf(temp * (x + 0.044715f * x * x * x)));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) 
{
    auto size_input = input.size();

    if (size_input == 0) 
    {
        return {};
    }

    float* d_input;
    float* d_output;
    std::vector<float> result(size_input);
    cudaMalloc(&d_input, size_input * sizeof(float));
    cudaMalloc(&d_output, size_input * sizeof(float));

    cudaMemcpy(d_input, input.data(), size_input * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size_input + threadsPerBlock - 1) / threadsPerBlock;
    geluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size_input);

    cudaMemcpy(result.data(), d_output, size_input * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}