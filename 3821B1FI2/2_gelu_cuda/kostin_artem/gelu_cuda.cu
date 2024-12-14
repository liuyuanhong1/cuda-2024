#include "gelu_cuda.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void gelu_cuda(const float* input, float* output, int s) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    constexpr float y = 0.797885f; // sqrt(2.0 / M_PI)
    constexpr float w = 0.0356774f; // y * 0.044715
    if (idx < s) {
        output[idx] = input[idx] * (1.0f / (1.0f + __expf(-2.0f * (input[idx] * (y + w * input[idx] * input[idx])))));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int s = input.size();
    std::vector<float> output(s);
    if (s == 0) return output;

    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);

    int length = s * sizeof(float);

    float *d_input, *d_output;
    cudaMalloc(&d_input, length);
    cudaMalloc(&d_output, length);

    cudaMemcpy(d_input, input.data(), length, cudaMemcpyHostToDevice);

    int blockSize = dev_prop.maxThreadsPerBlock;
    int gridSize = (s + blockSize - 1) / blockSize;

    gelu_cuda<<<gridSize, blockSize>>>(d_input, d_output, s);

    cudaMemcpy(output.data(), d_output, length, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
