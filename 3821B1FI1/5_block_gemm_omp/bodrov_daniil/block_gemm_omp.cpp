#include "block_gemm_omp.h"
#include <omp.h>
#include <vector>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    const int block_size = 192; // Оптимальный размер блока можно подобрать экспериментально

    #pragma omp parallel for
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int k = 0; k < n; k += block_size) {
                // Умножаем блоки: C_ij += A_ik * B_kj
                for (int ii = i; ii < std::min(i + block_size, n); ++ii) {
                    for (int kk = k; kk < std::min(k + block_size, n); ++kk) {
                        float a_ik = a[ii * n + kk];
                        // Развертывание внутреннего цикла по j для уменьшения накладных расходов
                        for (int jj = j; jj < std::min(j + block_size, n); jj += 4) {
                            c[ii * n + jj] += a_ik * b[kk * n + jj];
                            c[ii * n + jj + 1] += a_ik * b[kk * n + jj + 1];
                            c[ii * n + jj + 2] += a_ik * b[kk * n + jj + 2];
                            c[ii * n + jj + 3] += a_ik * b[kk * n + jj + 3];
                        }
                    }
                }
            }
        }
    }

    return c;
}