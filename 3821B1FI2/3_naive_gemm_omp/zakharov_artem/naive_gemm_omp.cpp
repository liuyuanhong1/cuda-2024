// Copyright (c) 2024 Zakharov Artem
#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n)
{
    std::vector<float> c(a.size(), 0);
    int shift = 0;
    int tmp = n;
    while (tmp != 1) {
        tmp >>= 1;
        shift++;
    }

    #pragma omp parallel for default(none) shared(a, b, c, n, shift)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                c[(i << shift) + j] += a[(i << shift) + k] * b[(k << shift) + j];
            }
        }
    }
    return c;
}
