// Copyright (c) 2024 Shmelev Ivan
#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& var1, const std::vector<float>& var2, int n) {
    int count_elem = n * n;
    if (var1.size() != count_elem || var2.size() != count_elem) return {};

    std::vector<float> var3(count_elem, 0.0f);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += var1[i * n + k] * var2[n * k + j];
            }
            var3[i * n + j] = sum;
        }
    }
    return var3;
}