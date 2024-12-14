#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    constexpr int BLOCK_SIZE = 64;
    auto size = n * n;
    std::vector<float> res(size, 0.0f);
    float sum;

#pragma omp parallel for collapse(3)
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            for (int k = 0; k < n; k += BLOCK_SIZE) {
                for (int ii = i; ii < i + BLOCK_SIZE && ii < n; ++ii) {
                    for (int jj = j; jj < j + BLOCK_SIZE && jj < n; ++jj) {
                        sum = 0.0f;
                        for (int kk = k; kk < k + BLOCK_SIZE && kk < n; ++kk) {
                            sum += a[ii * n + kk] * b[kk * n + jj];
                        }
                        res[ii * n + jj] += sum;
                    }
                }
            }
        }
    }
    return res;
}
