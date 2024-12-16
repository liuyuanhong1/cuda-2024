// Copyright (c) 2024 Khodyrev Fedor
#ifndef __BLOCK_GEMM_OMP_H
#define __BLOCK_GEMM_OMP_H

#include <vector>

std::vector<float> BlockGemmOMP(const std::vector<float>& var1, const std::vector<float>& var2, int size);

#endif  // __BLOCK_GEMM_OMP_H