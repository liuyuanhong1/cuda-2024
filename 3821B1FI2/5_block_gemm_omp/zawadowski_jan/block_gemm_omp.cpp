#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n) {
    std::vector<float> c(n * n, 0.0f);
    const int blockSize = 16;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i += blockSize) {
        for (int j = 0; j < n; j += blockSize) {
            for (int k = 0; k < n; k += blockSize) {
                for (int ii = i; ii < i + blockSize; ++ii) {
                    for (int jj = j; jj < j + blockSize; ++jj) {
                        float sum = 0.0f;
                        for (int kk = k; kk < k + blockSize; ++kk) {
                            sum += a[ii * n + kk] * b[kk * n + jj];
                        }
                        c[ii * n + jj] += sum;
                    }
                }
            }
        }
    }

    return c;
}