#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

#define CHECK_CUDA(call)                                                        \
    {                                                                           \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        }                                                                       \
    }

#define CHECK_CUBLAS(call)                                                      \
    {                                                                           \
        cublasStatus_t status = call;                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            throw std::runtime_error("cuBLAS error");                          \
        }                                                                       \
    }

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t bytes = n * n * sizeof(float);
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    CHECK_CUBLAS(cublasSetMatrix(n, n, sizeof(float), a.data(), n, d_A, n));
    CHECK_CUBLAS(cublasSetMatrix(n, n, sizeof(float), b.data(), n, d_B, n));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             n,
                             n,
                             n,
                             &alpha,
                             d_B, n,
                             d_A, n,
                             &beta,
                             d_C, n));

    std::vector<float> c(n * n);

    CHECK_CUBLAS(cublasGetMatrix(n, n, sizeof(float), d_C, n, c.data(), n));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return c;
}
