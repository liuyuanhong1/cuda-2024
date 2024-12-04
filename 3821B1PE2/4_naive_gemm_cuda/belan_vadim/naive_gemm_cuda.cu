#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <iostream>

#define CUDA_CHECK_ERROR(call)                                            \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            throw std::runtime_error(std::string("CUDA Error: ") +        \
                                    cudaGetErrorString(err));            \
        }                                                                 \
    } while (0)

__global__
void NaiveGemmKernel(const float* __restrict__ a,
                    const float* __restrict__ b_T,
                    float* __restrict__ c,
                    int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b_T[col * n + k];
        }
        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    if (a.size() != static_cast<size_t>(n * n) ||
        b.size() != static_cast<size_t>(n * n)) {
        throw std::invalid_argument("Input matrices must be of size n x n.");
    }

    std::vector<float> b_T(n * n);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            b_T[col * n + row] = b[row * n + col];
        }
    }

    float *d_a = nullptr, *d_b_T = nullptr, *d_c = nullptr;
    size_t bytes = n * n * sizeof(float);
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_b_T, bytes));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_c, bytes));

    CUDA_CHECK_ERROR(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_b_T, b_T.data(), bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                (n + blockDim.y - 1) / blockDim.y);

    NaiveGemmKernel<<<gridDim, blockDim>>>(d_a, d_b_T, d_c, n);

    CUDA_CHECK_ERROR(cudaGetLastError());

    std::vector<float> c(n * n);

    CUDA_CHECK_ERROR(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaFree(d_a));
    CUDA_CHECK_ERROR(cudaFree(d_b_T));
    CUDA_CHECK_ERROR(cudaFree(d_c));

    return c;
}
