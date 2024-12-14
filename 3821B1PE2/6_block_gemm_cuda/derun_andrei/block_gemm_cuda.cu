#include "block_gemm_cuda.h"

__global__ void blockGemmKernel(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c, int n)
{
    int blockSize = 16; // Block size (assuming 16x16 for simplicity)
    __shared__ float blockA[16][16];
    __shared__ float blockB[16][16];

    int row = blockIdx.x * blockSize + threadIdx.x;
    int col = blockIdx.y * blockSize + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float dotProduct = 0.0f;

    for (int i = 0; i < (n / blockSize); i++)
    {
        // Load A block into shared memory
        if (row < n && i * blockSize + ty < n)
        {
            blockA[ty][tx] = a[row * n + i * blockSize + tx];
        }
        else
        {
            blockA[ty][tx] = 0.0f; // Initialize to zero for boundary threads
        }

        // Load B block into shared memory
        if (col < n && i * blockSize + tx < n)
        {
            blockB[ty][tx] = b[(i * blockSize + ty) * n + col];
        }
        else
        {
            blockB[ty][tx] = 0.0f; // Initialize to zero for boundary threads
        }

        // Synchronize over all threads in block
        __syncthreads();

        // Compute BlockA * BlockB and accumulate into C block in shared memory
        for (int j = 0; j < blockSize; j++)
        {
            dotProduct += blockA[ty][j] * blockB[j][tx];
        }

        // Synchronize over all threads in block
        __syncthreads();
    }

    // Dump block C from shared to global memory
    if (row < n && col < n)
    {
        c[row * n + col] = dotProduct;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float> &a, const std::vector<float> &b, int n)
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
    int blockSize = 16;
    dim3 block(blockSize, blockSize);
    dim3 grid((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);
    blockGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);

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
