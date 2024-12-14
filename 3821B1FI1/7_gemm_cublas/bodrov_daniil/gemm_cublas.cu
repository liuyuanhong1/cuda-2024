#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <iostream>

// Класс для обработки ошибок CUDA и cuBLAS
class CUBLASError : public std::runtime_error {
public:
    explicit CUBLASError(const char* message)
        : std::runtime_error(message) {}
};

// Макрос для проверки ошибок CUDA
#define CHECK_CUDA(call)                                              \
    do {                                                             \
        cudaError_t err = (call);                                    \
        if (err != cudaSuccess) {                                    \
            throw CUBLASError(cudaGetErrorString(err));              \
        }                                                            \
    } while (0)

// Макрос для проверки ошибок cuBLAS
#define CHECK_CUBLAS(call)                                            \
    do {                                                             \
        cublasStatus_t status = (call);                              \
        if (status != CUBLAS_STATUS_SUCCESS) {                       \
            throw CUBLASError("cuBLAS operation failed");            \
        }                                                            \
    } while (0)

std::vector<float> GemmCUBLAS(const std::vector<float>& matrixA,
                                        const std::vector<float>& matrixB,
                                        int size) {
    // Результирующий вектор
    std::vector<float> matrixC(size * size);

    // Создание дескриптора cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Создание потока CUDA для асинхронных операций
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t bytes = size * size * sizeof(float);

    try {
        // Выделение памяти на устройстве
        CHECK_CUDA(cudaMalloc(&d_A, bytes));
        CHECK_CUDA(cudaMalloc(&d_B, bytes));
        CHECK_CUDA(cudaMalloc(&d_C, bytes));

        // Копирование данных с хоста на устройство (асинхронно)
        CHECK_CUBLAS(cublasSetMatrix(size, size, sizeof(float),
                                     matrixA.data(), size,
                                     d_A, size));
        CHECK_CUBLAS(cublasSetMatrix(size, size, sizeof(float),
                                     matrixB.data(), size,
                                     d_B, size));

        // Параметры умножения матриц
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Выполнение умножения матриц: C = A * B
        CHECK_CUBLAS(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 size, size, size,
                                 &alpha,
                                 d_B, size,
                                 d_A, size,
                                 &beta,
                                 d_C, size));

        // Копирование результата с устройства на хост (асинхронно)
        CHECK_CUBLAS(cublasGetMatrix(size, size, sizeof(float),
                                     d_C, size,
                                     matrixC.data(), size));

        // Синхронизация потока для завершения всех операций
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    catch (...) {
        // Освобождение ресурсов в случае исключения
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        cudaStreamDestroy(stream);
        cublasDestroy(handle);
        throw; // Переброс исключения
    }

    // Освобождение ресурсов
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(stream);
    cublasDestroy(handle);

    return matrixC;
}