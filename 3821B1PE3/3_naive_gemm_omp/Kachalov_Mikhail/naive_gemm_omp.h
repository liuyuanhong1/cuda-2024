// Copyright 2024 Kachalov Mikhail
#ifndef __NAIVE_GEMM_OMP_H
#define __NAIVE_GEMM_OMP_H

#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b,
                                int n);

#endif // __NAIVE_GEMM_OMP_H