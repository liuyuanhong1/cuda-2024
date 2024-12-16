#include "block_gemm_omp.h"
#include <omp.h>
#include <cstring>

std::vector<float> BlockGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n) {
    std::vector<float> c(n * n, 0.0f);

    constexpr int block_size = 16;

    if (n % block_size == 0) {

#pragma omp parallel for
        for (int block_i = 0; block_i < n; block_i += block_size) {
            for (int block_k = 0; block_k < n; block_k += block_size) {
           
                float a_block[block_size * block_size];
                for (int i = 0; i < block_size; i++) {
                    std::memcpy(a_block + i * block_size, a.data() + (block_i + i) * n + block_k, block_size * sizeof(float));
                }

                for (int block_j = 0; block_j < n; block_j += block_size) {

                    float b_block[block_size * block_size];
                    for (int k = 0; k < block_size; k++) {
                        std::memcpy(b_block + k * block_size, b.data() + (block_k + k) * n + block_j, block_size * sizeof(float));
                    }

                    for (int i = 0; i < block_size; i++) {
                        for (int k = 0; k < block_size; k++) {
                            for (int j = 0; j < block_size; j++) {
                                c[(block_i + i) * n + (block_j + j)] += a_block[i * block_size + k] * b_block[k * block_size + j];
                            }
                        }
                    }
                }
            }
        }
    }
    else {

#pragma omp parallel for
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                for (int k = 0; k < n; k++) {
                    c[row * n + col] += a[row * n + k] * b[k * n + col];
                }
            }
        }
    }

    return c;
}
