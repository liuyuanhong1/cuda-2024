// block_gemm_omp.cpp

#include "block_gemm_omp.h"
#include <omp.h>
#include <vector>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                         const std::vector<float>& b,
                                         int n) {
    std::vector<float> c(n * n, 0.0f);

    const int block_size = 64;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int k = 0; k < n; k += block_size) {
                for (int ii = i; ii < i + block_size; ++ii) {
                    for (int kk = k; kk < k + block_size; ++kk) {
                        float a_ik = a[ii * n + kk];
                        // Векторизация внутреннего цикла по j
                        #pragma omp simd
                        for (int jj = j; jj < j + block_size; ++jj) {
                            c[ii * n + jj] += a_ik * b[kk * n + jj];
                        }
                    }
                }
            }
        }
    }

    return c;
}