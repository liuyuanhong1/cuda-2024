// Copyright (c) 2024 Moiseev Nikita
#ifndef __NAIVE_GEMM_CUDA_H
#define __NAIVE_GEMM_CUDA_H

#include <vector>

std::vector<float> NaiveGemmCUDA(const std::vector<float>& matrix_a, const std::vector<float>& matrix_b, int dimension);

#endif // __NAIVE_GEMM_CUDA_H