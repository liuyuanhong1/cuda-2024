// Copyright (c) 2024 Volodin Evgeniy
#include "block_gemm_omp.h"
#include <stdexcept>
#include <omp.h>
#include <cstring>

std::vector<float> BlockGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (a.size() != n*n || b.size() != n*n) {
        throw std::invalid_argument("Matrix size does not match the specified n*n dimensions!");
    }

    int block_size = 16;

    std::vector<float> c(n * n, 0.0f);
    
    if (n % block_size == 0) {
        #pragma omp parallel for
        for (int br = 0; br < n; br += block_size) {
            for (int bc = 0; bc < n; bc += block_size) {
                float block_a[block_size * block_size];
                for (int i = 0; i < block_size; i++) {
                    memcpy(block_a + i * block_size, a.data() + (br + i) * n + bc, block_size * sizeof(float));
                }
                for (int bcc = 0; bcc < n; bcc += block_size) {
                    float block_b[block_size * block_size];
                    for (int k = 0; k < block_size; k++) {
                        memcpy(block_b + k * block_size, b.data() + (bc + k) * n + bcc, block_size * sizeof(float));
                    }
                    for (int i = 0; i < block_size; i++) {
                        for (int k = 0; k < block_size; k++) {
                            for (int j = 0; j < block_size; j++) {
                                c[(br + i) * n + (bcc + j)] +=
                                    block_a[i * block_size + k] * block_b[k * block_size + j];
                            }
                        }
                    }
                }
            }
        }
    }
    else {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                for (int j = 0; j < n; j++) {
                    c[i * n + j] += a[i * n + k] * b[k * n + j];
                }
            }
        }
    }
    return c;
}