#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Макрос для обработки ошибок CUDA
#define CUDA_ERROR(call)                                                \
    {                                                                   \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl;  \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    }

// Определяем размер блока
#define BLOCK_SIZE 16

// Оптимизированное CUDA ядро для блочного умножения матриц
__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    // Индексы строки и столбца текущего элемента
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Shared memory для блоков A и B
    __shared__ float A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B[BLOCK_SIZE][BLOCK_SIZE];

    float cVal = 0.0f;

    // Проходим по всех блоках по оси K
    for (int t = 0; t < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Векторизированная загрузка элементов матрицы A в shared memory
        if (row < n && (t * BLOCK_SIZE + threadIdx.x) < n)
            A[threadIdx.y][threadIdx.x] = a[row * n + t * BLOCK_SIZE + threadIdx.x];
        else
            A[threadIdx.y][threadIdx.x] = 0.0f;

        // Векторизированная загрузка элементов матрицы B в shared memory
        if (col < n && (t * BLOCK_SIZE + threadIdx.y) < n)
            B[threadIdx.y][threadIdx.x] = b[(t * BLOCK_SIZE + threadIdx.y) * n + col];
        else
            B[threadIdx.y][threadIdx.x] = 0.0f;

        // Синхронизация потоков внутри блока
        __syncthreads();

        // Разворот цикла для увеличения производительности
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            cVal += A[threadIdx.y][k] * B[k][threadIdx.x];
        }

        // Синхронизация перед загрузкой новых блоков
        __syncthreads();
    }

    // Запись результата в глобальную память
    if (row < n && col < n) {
        c[row * n + col] = cVal;
    }
}

// Host функция для запуска CUDA ядра
std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(n * n, 0.0f);

    size_t sizeInBytes = n * n * sizeof(float);

    // Выделяем память на устройстве
    float *d_a, *d_b, *d_c;
    CUDA_ERROR(cudaMalloc(&d_a, sizeInBytes));
    CUDA_ERROR(cudaMalloc(&d_b, sizeInBytes));
    CUDA_ERROR(cudaMalloc(&d_c, sizeInBytes));

    // Копируем данные на устройство
    CUDA_ERROR(cudaMemcpy(d_a, a.data(), sizeInBytes, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_b, b.data(), sizeInBytes, cudaMemcpyHostToDevice));

    // Определяем размеры сетки и блоков
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Запускаем CUDA ядро
    BlockGemmKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Проверяем наличие ошибок при запуске ядра
    CUDA_ERROR(cudaGetLastError());

    // Синхронизируем устройство
    CUDA_ERROR(cudaDeviceSynchronize());

    // Копируем результат обратно на хост
    CUDA_ERROR(cudaMemcpy(c.data(), d_c, sizeInBytes, cudaMemcpyDeviceToHost));

    // Освобождаем память на устройстве
    CUDA_ERROR(cudaFree(d_a));
    CUDA_ERROR(cudaFree(d_b));
    CUDA_ERROR(cudaFree(d_c));

    return c;
}