#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

__global__ void NaiveGemmKernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        
        float sum = 0.0;
        for (int k = 0; k < n; ++k) {
             sum += a[row * n + k] * b[k * n + col];
             printf("%f\n", sum);
        }
        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    std::vector<float> c(n * n, 0.0f);

    float* kernel_a = nullptr;
    float* kernel_b = nullptr;
    float* kernel_c = nullptr;

    cudaMalloc(&kernel_a, n * n * sizeof(float));
    cudaMalloc(&kernel_b, n * n * sizeof(float));
    cudaMalloc(&kernel_c, n * n * sizeof(float));
    cudaMemcpy(kernel_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(kernel_c, c.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    size_t size = 32;
    dim3 blockSize(size, size);
    dim3 numBlocks((n + size - 1) / size, (n + size - 1) / size);

    NaiveGemmKernel <<<numBlocks, blockSize >>> (kernel_a, kernel_b, kernel_c, n);
    cudaMemcpy(c.data(), kernel_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(kernel_a);
    cudaFree(kernel_b);
    cudaFree(kernel_c);

    return c;
}
