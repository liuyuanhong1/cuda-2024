#include "naive_gemm_cuda.h"

// CUDA Kernel for Naive Matrix Multiplication (GEMM)
__global__ void naiveGemmKernel(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n)
    {
        float dotProduct = 0.0f;
        for (int i = 0; i < n; i++)
        {
            // Coalesced global memory access for 'a' and 'b'
            dotProduct += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = dotProduct;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float> &a, const std::vector<float> &b, int n)
{
    if (n == 0 || a.size() != n * n || b.size() != n * n)
    {
        throw std::invalid_argument("Invalid input matrix dimensions");
    }

    // Allocate device memory
    float *d_a;
    float *d_b;
    float *d_c;
    cudaMalloc((void **)&d_a, n * n * sizeof(float));
    cudaMalloc((void **)&d_b, n * n * sizeof(float));
    cudaMalloc((void **)&d_c, n * n * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(d_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    int blockSize = 16; // Adjust based on your GPU's capabilities (e.g., 16x16, 32x32)
    dim3 block(blockSize, blockSize);
    dim3 grid((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);
    naiveGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    // Copy result from device to host
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
