// Copyright (c) 2024 Sokolova Daria
#include "naive_gemm_omp.h"

#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    long matrixSize = n * n;
    if (a.size() != matrixSize || b.size() != matrixSize) {
        return {};
    }

    std::vector<float> c(matrixSize, 0.0f);

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            for (int k = 0; k < n; ++k) {
                c[row * n + col] += a[row * n + k] * b[k * n + col];
            }
        }
    }

    return c;
}
