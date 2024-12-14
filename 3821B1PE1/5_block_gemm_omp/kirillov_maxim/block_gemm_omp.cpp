// Copyright (c) 2024 Kirillov Maxim
#include "block_gemm_omp.h"
#define SIZE 16

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    int blocksCount = n / SIZE;

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < blocksCount; ++bi) {
        for (int bj = 0; bj < blocksCount; ++bj) {
            for (int bk = 0; bk < blocksCount; ++bk) {
                for (int i = 0; i < SIZE; ++i) {
                    for (int j = 0; j < SIZE; ++j) {
                        float sum = 0.0f;
                        for (int k = 0; k < SIZE; ++k) {
                            sum += a[(bi * SIZE + i) * n + bk * SIZE + k] *
                                   b[(bk * SIZE + k) * n + bj * SIZE + j];
                        }
                        c[(bi * SIZE + i) * n + bj * SIZE + j] += sum;
                    }
                }
            }
        }
    }
    return c;
}