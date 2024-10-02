// Copyright (c) 2024 Kulikov Artem
#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b, int size) {
    std::vector<float> c(size * size, 0.0f);
#pragma omp parallel for shared(c)
    for (int m = 0; m < size; m++) {
        for (int n = 0; n < size; n++) {
            for (int k = 0; k < size; k++) {
                c[m * size + n] += a[m * size + k] * b[size * k + n];
            }
        }
    }
    return c;
}
