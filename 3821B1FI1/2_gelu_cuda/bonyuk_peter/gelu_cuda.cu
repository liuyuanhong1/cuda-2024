/*When entering the following vector:
std::vector<float> input = {1.0, 2.0, 3.0, 4.0, 5.0};
The output values ​​were:
{0.841192, 1.9546, 2.99636, 3.99993, 5 }
*/
#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#define CHECK_CUDA(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void gelu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float xCubed = x * x * x;
        float tanhArg = sqrt(2.0 / M_PI) * (x + 0.044715 * xCubed);
        float geluVal = 0.5f * x * (1.0f + tanh(tanhArg));
        output[idx] = geluVal;
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = input.size();
    std::vector<float> output(n);

    float* d_input;
    float* d_output;

    CHECK_CUDA(cudaMalloc((void**)&d_input, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output, n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    gelu_kernel<<<numBlocks, blockSize>>>(d_input, d_output, n);

    CHECK_CUDA(cudaMemcpy(output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return output;
}