// Copyright (c) 2024 Chuvashov Andrey
#include "naive_gemm_omp.h"

#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    if (a.size() != n * n || b.size() != n * n) {
        return {};
    }

    std::vector<float> c(n * n);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            float current = 0.0f;
            for (size_t k = 0; k < n; k++) {
                current += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = current;
        }
    }
    return c;
}
