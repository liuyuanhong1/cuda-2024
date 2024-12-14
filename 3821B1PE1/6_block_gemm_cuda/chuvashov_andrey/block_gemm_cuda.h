// Copyright (c) 2024 Chuvashov Andrey
#ifndef __BLOCK_GEMM_CUDA_H
#define __BLOCK_GEMM_CUDA_H

#define SIZE 16

#include <vector>

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n);

#endif // __BLOCK_GEMM_CUDA_H
