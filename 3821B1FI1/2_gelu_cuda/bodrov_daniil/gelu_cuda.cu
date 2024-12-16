/*Implement the function with the following interface in CUDA C++ using the formula described above:
```cpp
std::vector<float> GeluCUDA(const std::vector<float>& input);
```
Size of result vector should be the same as for `input`. Use CUDA technology to make your function work on NVIDIA GPU. Try to make it fast.
*/
// Including libs
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include "gelu_cuda.h"

// Error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(err); \
    } \
}

__global__ void gelu_cuda(float* input, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * powf(x, 3.0f))));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    if (n == 0) return {};

    float* d_input = nullptr;
    float* d_output = nullptr;
    size_t bytes = n * sizeof(float);

    // Allocate memory on the GPU
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    // Copy input data to GPU
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

    // Define kernel launch parameters
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    // Launch the kernel
    gelu_cuda<<<num_blocks, block_size>>>(d_input, d_output, n);

    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to CPU
    std::vector<float> output(n);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return output;
}