// Copyright (c) 2024 Khodyrev Fedor
#ifndef __NAIVE_GEMM_CUDA_H
#define __NAIVE_GEMM_CUDA_H

#include <vector>

std::vector<float> NaiveGemmCUDA(const std::vector<float>& var1, const std::vector<float>& var2, int n);

#endif // __NAIVE_GEMM_CUDA_H