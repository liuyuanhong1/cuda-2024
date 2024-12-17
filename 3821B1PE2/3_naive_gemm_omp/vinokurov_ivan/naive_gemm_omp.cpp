//  Copyright (c) 2024 Vinokurov Ivan
#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (a.size() != n * n || b.size() != n * n) {
        return {};
    }

    std::vector<float> output(n * n, 0.0f);

#pragma omp parallel for collapse(2)
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int inner = 0; inner < n; ++inner) {
                sum += a[row * n + inner] * b[inner * n + col];
            }
            output[row * n + col] = sum;
        }
    }

    return output;
}