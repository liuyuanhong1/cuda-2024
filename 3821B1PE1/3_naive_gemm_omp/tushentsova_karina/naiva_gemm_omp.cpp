// Copyright (c) 2024 Tushentsova Karina
#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    long total_elements = n * n;
    if (a.size() != total_elements || b.size() != total_elements) {
        return {};
    }

    std::vector<float> result(total_elements, 0.0f);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                result[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }

    return result;
}