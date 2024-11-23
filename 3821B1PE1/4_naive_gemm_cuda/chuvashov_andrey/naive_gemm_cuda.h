// Copyright (c) 2024 Chuvashov Andrey
#ifndef __NAIVE_GEMM_CUDA_H
#define __NAIVE_GEMM_CUDA_H

#include <vector>

#define SIZE 32

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n);

#endif // __NAIVE_GEMM_CUDA_H
