#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) 
{
    
    std::vector<float> c(n * n);

#pragma omp parallel for
    for (int i = 0; i < n; ++i) 
    {
        for (int j = 0; j < n; ++j) 
        {
            for (int r = 0; r < n; ++r) 
            {
                c[i * n + r] += a[i * n + j] * b[j * n + r];
            }
        }
    }

    return c;
}