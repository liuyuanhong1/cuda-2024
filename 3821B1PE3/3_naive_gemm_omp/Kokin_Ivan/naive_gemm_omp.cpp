// Copyright (c) 2024 Kokin Ivan

#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& mxA,
    const std::vector<float>& mxB,
    int src) {
    if (mxA.size() != src * src || mxB.size() != src * src) {
        return {};
    }

    std::vector<float> mxC(src * src, 0.0f);

#pragma omp parallel for collapse(2)
    for (int r = 0; r < src; ++r) {
        for (int c = 0; c < src; ++c) {
            float sum = 0.0f;
            for (int in = 0; in < src; ++in) {
                sum += mxA[r * src + in] * mxB[in * src + c];
            }
            mxC[r * src + c] = sum;
        }
    }

    return mxC;
}

