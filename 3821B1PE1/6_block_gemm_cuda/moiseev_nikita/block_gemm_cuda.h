// Copyright (c) 2024 Moiseev Nikita
#ifndef __BLOCK_GEMM_CUDA_H
#define __BLOCK_GEMM_CUDA_H

#include <vector>

std::vector<float> BlockGemmCUDA(const std::vector<float>& matrix_a, const std::vector<float>& matrix_b, int matrix_size);

#endif // __BLOCK_GEMM_CUDA_H
