#include <cuda_runtime.h>
#include "naive_gemm_cuda.h"

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

const int TILE_SIZE = 32;

__global__ void gemm_kernel_shared(const float* a, const float* b, float* c, int n) {
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < (n + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        if (row < n && (k * TILE_SIZE + threadIdx.x) < n) {
            tile_a[threadIdx.y][threadIdx.x] = a[row * n + (k * TILE_SIZE + threadIdx.x)];
        }
        else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < n && (k * TILE_SIZE + threadIdx.y) < n) {
            tile_b[threadIdx.y][threadIdx.x] = b[(k * TILE_SIZE + threadIdx.y) * n + col];
        }
        else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
    
        for (int t = 0; t < TILE_SIZE; ++t) {
            sum += tile_a[threadIdx.y][t] * tile_b[t][threadIdx.x];
        }
      
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    size_t size = n * n * sizeof(float);

    float* d_a;
    float* d_b;
    float* d_c;

    checkCudaError(cudaMalloc((void**)&d_a, size), "Failed to allocate memory for d_a");
    checkCudaError(cudaMalloc((void**)&d_b, size), "Failed to allocate memory for d_b");
    checkCudaError(cudaMalloc((void**)&d_c, size), "Failed to allocate memory for d_c");

    checkCudaError(cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice), "Failed to copy data for d_a");
    checkCudaError(cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice), "Failed to copy data for d_b");

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    gemm_kernel_shared << <numBlocks, threadsPerBlock >> > (d_a, d_b, d_c, n);
    checkCudaError(cudaGetLastError(), "Error when starting the kernel");

    checkCudaError(cudaDeviceSynchronize(), "Synchronization error");

    std::vector<float> c(n * n);
    checkCudaError(cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost), "Failed to copy data for c");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}