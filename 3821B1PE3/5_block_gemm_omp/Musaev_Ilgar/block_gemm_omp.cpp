#include "block_gemm_omp.h"
#include <omp.h>

void BlockMatrixMultiply(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, int n, int bs) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i += bs) {
        for (int j = 0; j < n; j += bs) {
            for (int k = 0; k < n; k += bs) {
                for (int ii = i; ii < i + bs && ii < n; ++ii) {
                    for (int jj = j; jj < j + bs && jj < n; ++jj) {
                        float sum = 0.0f;
                        for (int kk = k; kk < k + bs && kk < n; ++kk) {
                            sum += a[ii * n + kk] * b[kk * n + jj];
                        }
                        c[ii * n + jj] += sum;
                    }
                }
            }
        }
    }
}

std::vector<float> BlockGemmOMP(const std::vector<float>& a,const std::vector<float>& b,int n) {
    std::vector<float> c(n * n, 0.0f);
    int bs = 32;
    BlockMatrixMultiply(a, b, c, n, bs);
    return c;
}