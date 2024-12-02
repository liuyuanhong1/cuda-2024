#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <ctime>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

const float mdsqrt2ipi = -1.59577f;   // - 2 * sqrt(2 / PI)
const float c = 0.044715f;


__global__ void geluKernel(const float* input, float* output, size_t size)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        const float x = input[i];
        //GELU: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + c * x^3)))
        float result = x * (0.5f * (1.0f + tanh(mdsqrt2ipi * (x + c * x * x * x))));
        output[i] = result;
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
    

    geluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);

    cudaMemcpy(output.data(), d_output, sizeInBytes, cudaMemcpyDeviceToHost);


    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}