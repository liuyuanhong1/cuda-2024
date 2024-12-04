#include "block_gemm_omp.h"
#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    const int block_size = 64;
    std::vector<float> c(n * n, 0.0f);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += block_size) {
        for (int jj = 0; jj < n; jj += block_size) {
            for (int kk = 0; kk < n; kk += block_size) {
                for (int i = ii; i < std::min(ii + block_size, n); ++i) {
                    for (int k = kk; k < std::min(kk + block_size, n); ++k) {
                        float aik = a[i * n + k];
                        for (int j = jj; j < std::min(jj + block_size, n); ++j) {
                            c[i * n + j] += aik * b[k * n + j];
                        }
                    }
                }
            }
        }
    }

    return c;
}
