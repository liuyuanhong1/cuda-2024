#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << " at line "          \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
            exit(err);                                                        \
        }                                                                     \
    } while (0)

// Ядро для наивного умножения матриц
__global__ void gemmKernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

// Функция для умножения матриц с использованием CUDA
std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t bytes = n * n * sizeof(float);

    // Указатели для данных на устройстве
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    // Выделение памяти на устройстве и проверка ошибок
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // Копирование данных с хоста на устройство и проверка ошибок
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    // Определение параметров сетки и блоков
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Запуск ядра CUDA с проверкой на ошибки после выполнения
    gemmKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError()); // Проверка ошибок после запуска ядра
    CUDA_CHECK(cudaDeviceSynchronize()); // Синхронизация устройства для гарантии завершения

    // Копирование результата с устройства на хост
    std::vector<float> c(n * n);
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // Очистка памяти на устройстве
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return c;
}