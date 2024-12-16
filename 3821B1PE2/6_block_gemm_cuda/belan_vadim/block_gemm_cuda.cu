#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

#define CHECK_CUDA_ERROR(call)                                        \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__,    \
                    __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

__global__ void MatrixMulKernel(const float* A, const float* B, float* C, int n) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    int globalRow = blockRow * BLOCK_SIZE + threadRow;
    int globalCol = blockCol * BLOCK_SIZE + threadCol;

    float Cvalue = 0.0f;

    int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int m = 0; m < numTiles; ++m) {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        int A_row = globalRow;
        int A_col = m * BLOCK_SIZE + threadCol;
        int B_row = m * BLOCK_SIZE + threadRow;
        int B_col = globalCol;

        if (A_row < n && A_col < n)
            As[threadRow][threadCol] = A[A_row * n + A_col];
        else
            As[threadRow][threadCol] = 0.0f;

        if (B_row < n && B_col < n)
            Bs[threadRow][threadCol] = B[B_row * n + B_col];
        else
            Bs[threadRow][threadCol] = 0.0f;

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[threadRow][e] * Bs[e][threadCol];

        __syncthreads();
    }

    if (globalRow < n && globalCol < n)
        C[globalRow * n + globalCol] = Cvalue;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t size = n * n * sizeof(float);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, size));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, a.data(), size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, b.data(), size, cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Launching kernel with gridDim (%d, %d) and blockDim (%d, %d)\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    MatrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);

    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    std::vector<float> c(n * n);

    CHECK_CUDA_ERROR(cudaMemcpy(c.data(), d_C, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    return c;
}
