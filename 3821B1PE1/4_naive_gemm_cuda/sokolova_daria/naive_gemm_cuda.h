// Copyright (c) 2024 Sokolova Daria
#ifndef __NAIVE_GEMM_CUDA_H
#define __NAIVE_GEMM_CUDA_H

#include <vector>

#define BLOCK_SIZE 32

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n);

#endif // __NAIVE_GEMM_CUDA_H
