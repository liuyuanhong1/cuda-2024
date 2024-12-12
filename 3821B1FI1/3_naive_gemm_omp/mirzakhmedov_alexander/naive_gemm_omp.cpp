#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b, int size) {
    std::vector<float> c(size * size, 0.0f);
    float cnst;
    float* c_;
    const float* b_;
#pragma omp parallel for private(cnst, c_, b_)
    for (int m = 0; m < size; m++) {
        c_ = &c[m * size];
        for (int k = 0; k < size; k++) {
            cnst = a[m * size + k];
            b_ = &b[size * k];
            for (int n = 0; n < size; n+=2) {
                c_[n] += cnst * b_[n];
                c_[n+1] += cnst * b_[n+1];
            }
        }
    }
    return c;
}
