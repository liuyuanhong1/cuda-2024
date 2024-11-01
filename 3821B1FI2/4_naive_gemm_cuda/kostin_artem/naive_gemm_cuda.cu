#include "naive_gemm_cuda.h"
#include "cuda.h"
#include "cuda_runtime.h"

__global__ void MatrixMulKernel(const float* a, const float* b, float* res, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        res[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    auto size = n * n;
    std::vector<float> res(size, 0.0f);

    float *d_a, *d_b, *d_res;

    cudaMalloc(&d_a, size * sizeof(float));
    cudaMalloc(&d_b, size * sizeof(float));
    cudaMalloc(&d_res, size * sizeof(float));

    cudaMemcpy(d_a, a.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    MatrixMulKernel<<<gridSize, blockSize>>>(d_a, d_b, d_res, n);

    cudaMemcpy(res.data(), d_res, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    return res;
}
