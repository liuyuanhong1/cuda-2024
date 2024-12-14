// Copyright (c) 2024 Moiseev Nikita
#ifndef __BLOCK_GEMM_OMP_H
#define __BLOCK_GEMM_OMP_H

#include <vector>

std::vector<float> BlockGemmOMP(const std::vector<float>& matrix_a, const std::vector<float>& matrix_b, int matrix_size);

#endif  // __BLOCK_GEMM_OMP_H