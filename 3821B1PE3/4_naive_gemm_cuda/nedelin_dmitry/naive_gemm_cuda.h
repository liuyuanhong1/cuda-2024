// Copyright (c) 2024 Nedelin Dmitry

#ifndef __NAIVE_GEMM_CUDA_H
#define __NAIVE_GEMM_CUDA_H

#include <vector>

std::vector<float> NaiveGemmCUDA(const std::vector<float>& matrixA,
    const std::vector<float>& matrixB,
    int dimension);

#endif // __NAIVE_GEMM_CUDA_H