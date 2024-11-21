// Copyright (c) 2024 Khramov Ivan
#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    int elemCount = n * n;
    if (a.size() != elemCount || b.size() != elemCount) return {};

    std::vector<float> c(elemCount, 0.0f);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[n * k + j];
            }
            c[i * n + j] = sum;
        }
    }
    return c;
}
