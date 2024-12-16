// Copyright (c) 2024 Ivanov Nikita
#include "block_gemm_omp.h"
#include <omp.h>

void BlockMatrixMultiply(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, int n, int block_size) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int k = 0; k < n; k += block_size) {
                for (int ii = i; ii < i + block_size && ii < n; ++ii) {
                    for (int jj = j; jj < j + block_size && jj < n; ++jj) {
                        float sum = 0.0f;
                        for (int kk = k; kk < k + block_size && kk < n; ++kk) {
                            sum += a[ii * n + kk] * b[kk * n + jj];
                        }
                        c[ii * n + jj] += sum;
                    }
                }
            }
        }
    }
}

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    int block_size = 32;
    BlockMatrixMultiply(a, b, c, n, block_size);
    return c;
}
