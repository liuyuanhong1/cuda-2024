// Copyright (c) 2024 Vanushkin Dmitry
#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> result(n * n, 0.0f);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float x = a[i * n + j];
            for (int k = 0; k < n; ++k) {
                result[i * n + k] += x * b[j * n + k];
            }
        }
    }

    return result;
}
