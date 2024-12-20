#include "block_gemm_cuda.h"

void CUDA_CHECK(cudaError_t error) {
    if (error != cudaSuccess) { 
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at line " << __LINE__ << '\n'; 
        exit(EXIT_FAILURE); 
    }
}

#define BLOCK_SIZE 16

__global__ void block_gemm_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* C, int n) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int numBlocks = n / BLOCK_SIZE;
    float C_value = 0.0f;
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int k = 0; k < numBlocks; ++k) {
        As[threadIdx.y][threadIdx.x] = A[row * n + k * BLOCK_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(k * BLOCK_SIZE + threadIdx.y) * n + col];
        
        __syncthreads();
        for (int  m = 0; m < BLOCK_SIZE; ++m) {
            C_value += As[threadIdx.y][m] * Bs[m][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * n + col] = C_value;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    size_t bytes = n * n * sizeof(float);
    std::vector<float> c(n * n, 0.0f);
    float *d_a, *d_b, *d_c;

    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(n / BLOCK_SIZE, n / BLOCK_SIZE);
    block_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return c;
}