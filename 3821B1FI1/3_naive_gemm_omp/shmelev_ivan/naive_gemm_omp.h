// Copyright (c) 2024 Shmelev Ivan
#ifndef __NAIVE_GEMM_OMP_H
#define __NAIVE_GEMM_OMP_H

#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& var1, const std::vector<float>& var2, int n);

#endif  // __NAIVE_GEMM_OMP_H