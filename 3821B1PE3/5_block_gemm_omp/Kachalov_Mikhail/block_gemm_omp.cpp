// Copyright 2024 Kachalov Mikhail
#include "block_gemm_omp.h"
#include <omp.h>
#include <vector>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b,
                                int n)
{
    const int blockSize = 64;
    std::vector<float> c(n * n, 0.0f);

#pragma omp parallel for collapse(2)
    for (int bi = 0; bi < n; bi += blockSize)
    {
        for (int bj = 0; bj < n; bj += blockSize)
        {
            for (int bk = 0; bk < n; bk += blockSize)
            {
                for (int i = bi; i < std::min(bi + blockSize, n); ++i)
                {
                    for (int j = bj; j < std::min(bj + blockSize, n); ++j)
                    {
                        float sum = 0.0f;
                        for (int k = bk; k < std::min(bk + blockSize, n); ++k)
                        {
                            sum += a[i * n + k] * b[k * n + j];
                        }
                        c[i * n + j] += sum;
                    }
                }
            }
        }
    }
    return c;
}