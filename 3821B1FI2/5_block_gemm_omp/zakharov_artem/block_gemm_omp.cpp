// Copyright (c) 2024 Zakharov Artem
#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(a.size(), 0);
    int block_size = std::min(n, 4);
    int num_blocks_in_line = n / block_size;

    #pragma omp parallel for default(none) shared(c, a, b, n, block_size, num_blocks_in_line) collapse(2)
    for (int block_i = 0; block_i < num_blocks_in_line; block_i++) {
        for (int block_j = 0; block_j < num_blocks_in_line; block_j++) {
            for (int k = 0; k < num_blocks_in_line; k++) {
                for (int i = block_i * block_size; i < (block_i + 1) * block_size; i++) {
                    for (int j = block_j * block_size; j < (block_j + 1) * block_size; j++) {
                        for (int q = k  * block_size; q < (k + 1) * block_size; q++) {
                            c[i * n + j] += a[i * n + q] * b[q * n + j];
                        }
                    }
                }
            }
        }
    }
    return c;
}
