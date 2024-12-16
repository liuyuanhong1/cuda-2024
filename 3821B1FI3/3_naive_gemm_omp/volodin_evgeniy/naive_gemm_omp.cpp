// Copyright (c) 2024 Volodin Evgeniy
#include "naive_gemm_omp.h"
#include <omp.h>
#include <stdexcept>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (a.size() != n * n || b.size() != n * n) {
        throw std::invalid_argument("Matrix size does not match the specified n*n dimensions");
    }

    std::vector<float> c(n*n, 0.0f);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                c[i*n+j] += a[i*n+k] * b[k*n+j];
            }
        }
    }

    return c;
}