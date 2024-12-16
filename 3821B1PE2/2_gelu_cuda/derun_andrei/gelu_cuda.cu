#include "gelu_cuda.h"

// CUDA Kernel for GELU computation
__global__ void geluKernel(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float x = input[idx];
        float gelu = 0.5f * x * (1 + tanh(sqrtf(2 / 3.14159f) * (x + 0.044715f * x * x)));
        output[idx] = gelu;
    }
}

std::vector<float> GeluCUDA(const std::vector<float> &input)
{
    int size = input.size();
    if (size == 0)
        return {}; // edge case: empty input

    // Allocate device memory
    float *d_input;
    float *d_output;
    cudaMalloc((void **)&d_input, size * sizeof(float));
    cudaMalloc((void **)&d_output, size * sizeof(float));

    // Copy input from host to device
    cudaMemcpy(d_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    int blockSize = 32;                                 // adjust based on your GPU's capabilities
    int numBlocks = (size + blockSize - 1) / blockSize; // ceiling division
    geluKernel<<<numBlocks, blockSize>>>(d_input, d_output, size);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    // Copy output from device to host
    std::vector<float> output(size);
    cudaMemcpy(output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}