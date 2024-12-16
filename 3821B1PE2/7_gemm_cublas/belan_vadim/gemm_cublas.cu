#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call)                                                          \
    {                                                                             \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl;                    \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    }

#define CHECK_CUBLAS(call)                                                         \
    {                                                                              \
        cublasStatus_t err = call;                                                 \
        if (err != CUBLAS_STATUS_SUCCESS) {                                        \
            std::cerr << "CUBLAS error at " << __FILE__ << ":" << __LINE__ << " - "\
                      << err << std::endl;                                         \
            std::exit(EXIT_FAILURE);                                               \
        }                                                                          \
    }

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {

    std::vector<float> A_col_major(n * n);
    std::vector<float> B_col_major(n * n);
    std::vector<float> C_col_major(n * n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_col_major[j * n + i] = a[i * n + j];
            B_col_major[j * n + i] = b[i * n + j];
        }
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_A, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, n * n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, A_col_major.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B_col_major.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;

    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n, n, n,
                             &alpha,
                             d_A, n,
                             d_B, n,
                             &beta,
                             d_C, n));

    CHECK_CUDA(cudaMemcpy(C_col_major.data(), d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> c(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[i * n + j] = C_col_major[j * n + i];
        }
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return c;
}
