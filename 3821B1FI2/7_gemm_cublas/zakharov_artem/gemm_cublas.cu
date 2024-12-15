// Copyright (c) 2024 Zakharov Artem
#include "gemm_cublas.h"
#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    int size = n * n;
    std::vector<float> c(size);
    size_t bytes_size = size * sizeof(float);
    float alpha = 1.0;
    float beta = 0.0;

    float *a_dev, *b_dev, *c_dev;

    cudaMalloc(reinterpret_cast<void**>(&a_dev), bytes_size);
    cudaMalloc(reinterpret_cast<void**>(&b_dev), bytes_size);
    cudaMalloc(reinterpret_cast<void**>(&c_dev), bytes_size);

    cudaMemcpy(reinterpret_cast<void*>(a_dev),
               reinterpret_cast<const void*>(a.data()),
               bytes_size, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(b_dev),
               reinterpret_cast<const void*>(b.data()),
               bytes_size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha,
                b_dev, size, a_dev, size, &beta, c_dev, size);
    cublasDestroy(handle);

    cudaMemcpy(reinterpret_cast<void*>(c.data()),
               reinterpret_cast<const void*>(c_dev),
               bytes_size, cudaMemcpyDeviceToHost);

    cudaFree(reinterpret_cast<void*>(a_dev));
    cudaFree(reinterpret_cast<void*>(b_dev));
    cudaFree(reinterpret_cast<void*>(c_dev));

    return c;
}
