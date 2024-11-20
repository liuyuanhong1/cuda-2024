#include "block_gemm_omp.h"
#include <omp.h>
#include <cmath>

// Реализация функции для блочного перемножения двух квадратных матриц
std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    const int block_size = 32; // Размер блока (можно настроить)
    std::vector<float> c(n * n, 0.0f); // Результирующая матрица

    // Параллельный цикл с использованием OpenMP
#pragma omp parallel for collapse(2)
    for (int bi = 0; bi < n; bi += block_size) {
        for (int bj = 0; bj < n; bj += block_size) {
            for (int bk = 0; bk < n; bk += block_size) {
                // Обработка блоков
                for (int i = bi; i < std::min(bi + block_size, n); ++i) {
                    for (int j = bj; j < std::min(bj + block_size, n); ++j) {
                        float sum = 0.0f;
                        for (int k = bk; k < std::min(bk + block_size, n); ++k) {
                            sum += a[i * n + k] * b[k * n + j];
                        }
                        c[i * n + j] += sum;
                    }
                }
            }
        }
    }

    return c;
}
