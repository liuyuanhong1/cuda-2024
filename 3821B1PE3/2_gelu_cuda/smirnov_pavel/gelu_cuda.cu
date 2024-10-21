#include "gelu_cuda.h"
#include <cuda_fp16.h>

#define BLOCK_SIZE 256

__global__ void geluKernel(const half* input, half* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float x3 = x * x * x;
        float tanh_arg = sqrtf(2.0f / M_PI) * (x + 0.044715f * x3);
        output[idx] = 0.5f * x * (1.0f + tanhf(tanh_arg));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t size = input.size();
    std::vector<float> output(size);

    half* d_input;
    half* d_output;

    cudaMalloc(&d_input, size * sizeof(half));
    cudaMalloc(&d_output, size * sizeof(half));

    cudaMemcpy(d_input, input.data(), size * sizeof(half), cudaMemcpyHostToDevice);

    const int threadsPerBlock = BLOCK_SIZE;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock; 
    geluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);

    cudaDeviceSynchronize(); 

    cudaMemcpy(output.data(), d_output, size * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
