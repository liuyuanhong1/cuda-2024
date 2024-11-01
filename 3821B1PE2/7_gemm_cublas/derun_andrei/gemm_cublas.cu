#include "gemm_cublas.h"

void CheckCudaStatus(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime API error %d: %s\n", status, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

void CheckCublasStatus(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cuBLAS API error %d\n", status);
        exit(EXIT_FAILURE);
    }
}

std::vector<float> GemmCUBLAS(const std::vector<float> &a,
                              const std::vector<float> &b,
                              int n)
{
    assert(a.size() == n * n && b.size() == n * n && "Matrix size mismatch");

    float *d_A, *d_B, *d_C;
    CheckCudaStatus(cudaMalloc((void **)&d_A, n * n * sizeof(float)));
    CheckCudaStatus(cudaMalloc((void **)&d_B, n * n * sizeof(float)));
    CheckCudaStatus(cudaMalloc((void **)&d_C, n * n * sizeof(float)));

    CheckCudaStatus(cudaMemcpy(d_A, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    CheckCudaStatus(cudaMemcpy(d_B, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CheckCublasStatus(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CheckCublasStatus(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n));

    std::vector<float> h_C(n * n);

    CheckCudaStatus(cudaMemcpy(h_C.data(), d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    CheckCudaStatus(cudaFree(d_A));
    CheckCudaStatus(cudaFree(d_B));
    CheckCudaStatus(cudaFree(d_C));
    CheckCublasStatus(cublasDestroy(handle));

    return h_C;
}
