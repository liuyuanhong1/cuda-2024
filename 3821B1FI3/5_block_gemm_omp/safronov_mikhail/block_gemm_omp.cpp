#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    std::vector<float> c(n * n);
    constexpr int block_sz = 16;
    if (n % block_sz == 0) {
#pragma omp parallel for
        for (int ii = 0; ii < n; ii += block_sz) {
            for (int kk = 0; kk < n; kk += block_sz) {
                float block_a[block_sz * block_sz];
                for (int i = 0; i < block_sz; i++) {
                    memcpy(block_a + i * block_sz, a.data() + (ii + i) * n + kk, block_sz * sizeof(float));
                }
                for (int jj = 0; jj < n; jj += block_sz) {
                    float block_b[block_sz * block_sz];
                    for (int k = 0; k < block_sz; k++) {
                        memcpy(block_b + k * block_sz, b.data() + (kk + k) * n + jj, block_sz * sizeof(float));
                    }
                    for (int i = 0; i < block_sz; i++) {
                        for (int k = 0; k < block_sz; k++) {
                            for (int j = 0; j < block_sz; j++) {
                                c[(ii + i) * n + (jj + j)] +=
                                    block_a[i * block_sz + k] * block_b[k * block_sz + j];
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
