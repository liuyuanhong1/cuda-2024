// Copyright (c) 2024 Vinichuk Timofey
#include "block_gemm_omp.h"
#include <omp.h>
#include <cmath>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    int block_size = 32;
    std::vector<float> c(n * n, 0.0f);

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

    return c;
}