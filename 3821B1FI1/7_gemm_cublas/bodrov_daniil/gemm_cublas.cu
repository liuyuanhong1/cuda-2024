#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
    }

// Макрос для проверки ошибок cuBLAS
#define CUBLAS_CHECK(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error("cuBLAS Error: " + std::to_string(err)); \
    }

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    // Размер матриц в байтах
    size_t bytes = n * n * sizeof(float);

    // Указатели на устройства
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    // Выделение памяти на устройстве
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    // Копирование данных с хоста на устройство
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    // Создание дескриптора cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Параметры для умножения матриц
    // В cuBLAS матрицы хранятся в столбцовом порядке, поэтому
    // чтобы работать с матрицами в строковом порядке, мы используем транспонирование
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Выполнение операции C = A * B
    // В cuBLAS: C = alpha * op(A) * op(B) + beta * C
    // Для строкового порядка: C^T = B^T * A^T
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_T,
                             n, n, n,
                             &alpha,
                             d_b, n,
                             d_a, n,
                             &beta,
                             d_c, n));

    // Выделение памяти для результата на хосте
    std::vector<float> c(n * n);

    // Копирование результата с устройства на хост
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return c;
}