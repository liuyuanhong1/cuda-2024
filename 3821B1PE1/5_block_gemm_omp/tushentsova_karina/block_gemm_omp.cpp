// Copyright (c) 2024 Tushentsova Karina
#include "block_gemm_omp.h"
#include <omp.h>

#define SIZE_BLOCK 16

std::vector<float> BlockGemmOMP(const std::vector<float>& a, const std::vector<float>& b,int n) {
    std::vector<float> result(n * n, 0.0f);

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int block_row = 0; block_row < n; block_row += SIZE_BLOCK) {
        for (int block_col = 0; block_col < n; block_col += SIZE_BLOCK) {
            for (int block_k = 0; block_k < n; block_k += SIZE_BLOCK) {
                for (int row = block_row; row < block_row + SIZE_BLOCK && row < n; ++row) {
                    for (int col = block_col; col < block_col + SIZE_BLOCK && col < n; ++col) {
                        float sum = 0.0f;

                        for (int k = block_k; k < block_k + SIZE_BLOCK && k < n; ++k) {
                            sum += a[row * n + k] * b[k * n + col];
                        }
                        result[row * n + col] += sum;
                    }
                }
            }
        }
    }

    return result;
}