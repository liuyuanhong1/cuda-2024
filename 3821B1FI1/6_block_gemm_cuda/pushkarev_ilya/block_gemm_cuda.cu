#include <cstdlib>
#include <cuda.h>

#include "block_gemm_cuda.h"
#include "cuda_runtime.h"

#define BLOCK_SIZE 32

__global__ void BlockGemmKernel(const float* a, const float* b,
                                float* const c, const int size) 
{

    __shared__ float aCached[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float bCached[BLOCK_SIZE][BLOCK_SIZE];

    const int threadX = threadIdx.x;
    const int threadY = threadIdx.y;

    const int rowIndex = blockIdx.y * BLOCK_SIZE + threadY;
    const int colIndex = blockIdx.x * BLOCK_SIZE + threadX;

    float resultValue = 0.0f;

    for (int tile = 0; tile < size / BLOCK_SIZE; ++tile) 
    {
        aCached[threadY][threadX] = a[rowIndex * size + tile * BLOCK_SIZE + threadX];
        bCached[threadY][threadX] = b[(tile * BLOCK_SIZE + threadY) * size + colIndex];

        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            resultValue += aCached[threadY][k] * bCached[k][threadX];
        }
        __syncthreads();
    }

    if (rowIndex < size && colIndex < size) 
    {
        c[rowIndex * size + colIndex] = resultValue;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, 
                                 int n) 
{
    std::vector<float> resultMatrix(n * n);

    size_t sizeInBytes = n * n * sizeof(*a.data());

    float* deviceMatrixA;
    cudaMalloc(&deviceMatrixA, sizeInBytes);
    float* deviceMatrixB;
    cudaMalloc(&deviceMatrixB, sizeInBytes);
    float* deviceResultMatrix;
    cudaMalloc(&deviceResultMatrix, sizeInBytes);

    cudaMemcpy(deviceMatrixA, a.data(), sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, b.data(), sizeInBytes, cudaMemcpyHostToDevice);

    const int blockSizeAxis = BLOCK_SIZE;
    dim3 threadsPerBlock(blockSizeAxis, blockSizeAxis);
    dim3 numBlocks((n + blockSizeAxis - 1) / blockSizeAxis, (n + blockSizeAxis - 1) / blockSizeAxis);

    BlockGemmKernel<<<numBlocks, threadsPerBlock>>>(deviceMatrixA, deviceMatrixB, deviceResultMatrix, n);

    cudaMemcpy(resultMatrix.data(), deviceResultMatrix, sizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceResultMatrix);
    return resultMatrix;
}