#include "block_gemm_omp.h"
#include <omp.h>
#include <vector>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    const int blockSize = 32; // Размер блока (можно изменить для оптимизации)
    std::vector<float> c(n * n, 0.0f);

// Multiplying matrix
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i += blockSize) {
        for (int j = 0; j < n; j += blockSize) {
            for (int k = 0; k < n; k += blockSize) {
                for (int ii = i; ii < i + blockSize && ii < n; ++ii) {
                    for (int jj = j; jj < j + blockSize && jj < n; ++jj) {
                        float sum = 0.0f;
                        for (int kk = k; kk < k + blockSize && kk < n; ++kk) {
                            sum += a[ii * n + kk] * b[kk * n + jj];
                        }
                        c[ii * n + jj] += sum;
                    }
                }
            }
        }
    }

    return c;
}
