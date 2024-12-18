// Copyright (c) 2024 Soloninko Andrey
#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b, int size) {

    auto countElem = size * size;
    
    std::vector<float> c(countElem, 0.0f);

#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < size; ++k) {
            float temp = a[i * size + k];
            for (int j = 0; j < size; ++j) {
                c[i * size + j] += temp * b[k * size + j];
            }
        }
    }

    return c;
}
