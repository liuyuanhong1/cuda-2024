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

// CUDA kernel для оптимизированного блочного умножения матриц
__global__ void BlockGemmKernel(const float* a, const float* b, float* c, int n, int block_size) {
    // Индексы строки и столбца текущего элемента
    int row = blockIdx.y * block_size + threadIdx.y;
    int col = blockIdx.x * block_size + threadIdx.x;

    // Shared memory для блоков A и B
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_size * block_size;

    float sum = 0.0f;

    // Проходим по всем блокам
    for (int k = 0; k < n; k += block_size) {
        // Загружаем блоки A и B в shared memory
        if (row < n && (k + threadIdx.x) < n) {
            As[threadIdx.y * block_size + threadIdx.x] = a[row * n + (k + threadIdx.x)];
        } else {
            As[threadIdx.y * block_size + threadIdx.x] = 0.0f;
        }

        if ((k + threadIdx.y) < n && col < n) {
            Bs[threadIdx.y * block_size + threadIdx.x] = b[(k + threadIdx.y) * n + col];
        } else {
            Bs[threadIdx.y * block_size + threadIdx.x] = 0.0f;
        }

        // Синхронизируем потоки внутри блока
        __syncthreads();

        // Развертывание цикла для увеличения производительности
        #pragma unroll
        for (int e = 0; e < block_size; ++e) {
            sum += As[threadIdx.y * block_size + e] * Bs[e * block_size + threadIdx.x];
        }

        // Синхронизируем перед загрузкой новых блоков
        __syncthreads();
    }

    // Записываем результат в глобальную память
    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

// Host функция для запуска CUDA ядра
std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const int block_size = 32; // Оптимальное значение для использования возможностей GPU
    size_t bytes = n * n * sizeof(float);

    // Выделяем память на устройстве
    float *d_a, *d_b, *d_c;
    CUDA_ERROR(cudaMalloc(&d_a, bytes));
    CUDA_ERROR(cudaMalloc(&d_b, bytes));
    CUDA_ERROR(cudaMalloc(&d_c, bytes));

    // Копируем данные на устройство
    CUDA_ERROR(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    // Определяем размеры сетки и блоков
    dim3 threads(block_size, block_size);
    dim3 grid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);
    size_t shared_mem_size = 2 * block_size * block_size * sizeof(float);

    // Запускаем CUDA ядро
    BlockGemmKernel<<<grid, threads, shared_mem_size>>>(d_a, d_b, d_c, n, block_size);
    CUDA_ERROR(cudaDeviceSynchronize());

    // Копируем результат обратно на хост
    std::vector<float> c(n * n);
    CUDA_ERROR(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // Освобождаем память на устройстве
    CUDA_ERROR(cudaFree(d_a));
    CUDA_ERROR(cudaFree(d_b));
    CUDA_ERROR(cudaFree(d_c));

    return c;
}