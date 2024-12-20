#include "naive_gemm_cuda.h"
#include "cuda_runtime.h"


__global__ void naiveGemmKernel(const float* a, const float* b, float* c, int n) 
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) 
    {
        float sum = 0.0f;
        for (int r = 0; r < n; ++r) 
        {
            sum += a[i * n + r] * b[r * n + j];
        }
        c[i * n + j] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) 
{
    std::vector<float> c(n * n, 0.0f);

    float* d_a;
    float* d_b;
    float* d_c;

    cudaMalloc(&d_a, n * n * sizeof(float));
    cudaMalloc(&d_b, n * n * sizeof(float));
    cudaMalloc(&d_c, n * n * sizeof(float));
    cudaMemcpy(d_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    const size_t blockSize = 32u;
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 gridDimensions((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);

    naiveGemmKernel<<<gridDimensions, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}