#include <cuda.h>

#include "gemm_cublas.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {


    const float q = 1.0f;
    const float p = 0.0f;

    std::vector<float> c(n * n);

    float* deviceA;
    float* deviceB;
    float* deviceC;

    cudaMalloc(&deviceA, sizeof(float) * n * n);
    cudaMalloc(&deviceB, sizeof(float) * n * n);
    cudaMalloc(&deviceC, sizeof(float) * n * n);

    cudaMemcpy(deviceA, a.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice);

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &q, deviceB, n, deviceA, n, &p, deviceC, n);

    cublasDestroy(cublasHandle);

    cudaMemcpy(c.data(), deviceC, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return c;
}