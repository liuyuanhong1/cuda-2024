// Copyright (c) 2024 Morgachev Stepan
#include "naive_gemm_omp.h"

#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    long size = n * n;
    if (a.size() != size || b.size() != size) {
        return {};
    }

    std::vector<float> result(size, 0.0f);

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
