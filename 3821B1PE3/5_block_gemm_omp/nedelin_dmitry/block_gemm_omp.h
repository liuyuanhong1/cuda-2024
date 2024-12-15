// Copyright (c) 2024 Nedelin Dmitry
#ifndef __BLOCK_GEMM_OMP_H
#define __BLOCK_GEMM_OMP_H

#include <vector>

std::vector<float> BlockGemmOMP(const std::vector<float>& matrixA,
    const std::vector<float>& matrixB,
    int dimension);

#endif // __BLOCK_GEMM_OMP_H
