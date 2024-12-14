#include "block_gemm_cuda.h"
#include "cuda.h"
#include "cuda_runtime.h"

constexpr int BLOCK_SIZE = 32;

__global__ void BlockGemmKernel(const float* __restrict__ a, const float* __restrict__ b, float* c, int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int row = threadIdx.y;
    int col = threadIdx.x;

    int globalRow = blockRow * BLOCK_SIZE + row;
    int globalCol = blockCol * BLOCK_SIZE + col;

    float sum = 0.0f;

    for (int m = 0; m < n / BLOCK_SIZE; ++m) {
        As[row][col] = a[globalRow * n + (m * BLOCK_SIZE + col)];
        Bs[row][col] = b[(m * BLOCK_SIZE + row) * n + globalCol];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[row][k] * Bs[k][col];
        }

        __syncthreads();
    }

    c[globalRow * n + globalCol] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    int size = n * n;
    std::vector<float> c(size, 0.0f);

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, size * sizeof(float));
    cudaMalloc(&d_b, size * sizeof(float));
    cudaMalloc(&d_c, size * sizeof(float));

    cudaMemcpy(d_a, a.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(n / BLOCK_SIZE, n / BLOCK_SIZE);

    BlockGemmKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
