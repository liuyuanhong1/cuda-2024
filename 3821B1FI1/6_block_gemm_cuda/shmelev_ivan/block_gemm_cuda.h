// Copyright (c) 2024 Shmelev Ivan
#ifndef __BLOCK_GEMM_CUDA_H
#define __BLOCK_GEMM_CUDA_H

#include <vector>

std::vector<float> BlockGemmCUDA(const std::vector<float>& var1, const std::vector<float>& var2, int n);

#endif // __BLOCK_GEMM_CUDA_H