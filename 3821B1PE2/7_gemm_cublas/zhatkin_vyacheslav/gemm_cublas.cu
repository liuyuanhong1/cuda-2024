#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

// Функция для транспонирования матрицы
__global__ void transpose(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n && idy < n) {
        out[idy * n + idx] = in[idx * n + idy];
    }
}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    // Инициализация cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Выделяем память на устройстве
    float *d_a, *d_b, *d_c, *d_a_transposed, *d_b_transposed;
    cudaMalloc((void**)&d_a, n * n * sizeof(float));  // Матрица A (n x n)
    cudaMalloc((void**)&d_b, n * n * sizeof(float));  // Матрица B (n x n)
    cudaMalloc((void**)&d_c, n * n * sizeof(float));  // Матрица C (n x n)
    cudaMalloc((void**)&d_a_transposed, n * n * sizeof(float)); // Транспонированная A
    cudaMalloc((void**)&d_b_transposed, n * n * sizeof(float)); // Транспонированная B

    // Копируем данные с хоста на устройство
    cudaMemcpy(d_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Транспонируем матрицы A и B на устройстве
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);
    transpose<<<gridSize, blockSize>>>(d_a_transposed, d_a, n);
    transpose<<<gridSize, blockSize>>>(d_b_transposed, d_b, n);
    cudaDeviceSynchronize();

    // Выполняем умножение матриц: C = A * B
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Теперь данные в d_a_transposed и d_b_transposed находятся в правильном формате для cuBLAS
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_a_transposed, n,  // A транспонирована, это n x n в столбцовом порядке
                d_b_transposed, n,  // B транспонирована, это n x n в столбцовом порядке
                &beta,
                d_c, n);  // C будет n x n

    // Копируем результат с устройства на хост
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Освобождаем память
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_a_transposed);
    cudaFree(d_b_transposed);
    cublasDestroy(handle);

    // Возвращаем результат (в формате row-major)
    return c;
}