#include "block_gemm_cuda.h"
#include <cassert>
#include <cuda_runtime.h>

static constexpr int BLOCK_SIZE = 32;
static constexpr int THREADS_PER_DIM = 32;


__global__ void BlockGemmCUDA_kernel(const float* a, const float* b, float* c, const int n) {
    __shared__ float a_block[THREADS_PER_DIM][THREADS_PER_DIM];
    __shared__ float b_block[THREADS_PER_DIM][THREADS_PER_DIM];


    const int i = blockIdx.y * THREADS_PER_DIM + threadIdx.y;
    const int j = blockIdx.x * THREADS_PER_DIM + threadIdx.x;

    float result = 0.0f;


    for (int k_block = 0; k_block < n / THREADS_PER_DIM; k_block++) {

        a_block[threadIdx.y][threadIdx.x] = a[i * n + (k_block * THREADS_PER_DIM + threadIdx.x)];
        b_block[threadIdx.y][threadIdx.x] = b[(k_block * THREADS_PER_DIM + threadIdx.y) * n + j];
        

        __syncthreads();


        for (int k = 0; k < THREADS_PER_DIM; k++) {
            result += a_block[threadIdx.y][k] * b_block[k][threadIdx.x];
        }


        __syncthreads();
    }


    if (i < n && j < n) {
        c[i * n + j] = result;
    }
}


std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {


    assert(n % THREADS_PER_DIM == 0);

    const int gridSize = n / THREADS_PER_DIM;
    const dim3 blocks(gridSize, gridSize);
    const dim3 threads(THREADS_PER_DIM, THREADS_PER_DIM);


    std::vector<float> c(n * n);

    float* d_a;
    float* d_b;
    float* d_c;


    cudaMalloc(&d_a, sizeof(float) * n * n);
    cudaMalloc(&d_b, sizeof(float) * n * n);
    cudaMalloc(&d_c, sizeof(float) * n * n);


    cudaMemcpy(d_a, a.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice);


    BlockGemmCUDA_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);


    cudaDeviceSynchronize();


    cudaMemcpy(c.data(), d_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost);


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
