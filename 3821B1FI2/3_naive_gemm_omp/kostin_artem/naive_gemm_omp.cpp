#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    auto size = n * n;
    std::vector<float> res(size, 0.0f);
#pragma omp parallel for collapse(2)
    for (auto i = 0; i < n; ++i) {
        for (auto j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (auto k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            res[i * n + j] += sum;
        }
    }
    return res;
}
