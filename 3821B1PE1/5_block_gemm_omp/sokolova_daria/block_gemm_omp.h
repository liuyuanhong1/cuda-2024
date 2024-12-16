// Copyright (c) 2024 Sokolova Daria
#ifndef __BLOCK_GEMM_OMP_H
#define __BLOCK_GEMM_OMP_H

#include <vector>

#define BLOCK_SIZE 16

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n);

#endif // __BLOCK_GEMM_OMP_H
