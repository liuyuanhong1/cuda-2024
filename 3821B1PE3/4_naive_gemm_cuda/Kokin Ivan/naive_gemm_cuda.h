// Copyright (c) 2024 Kokin Ivan

#ifndef __NAIVE_GEMM_CUDA_H
#define __NAIVE_GEMM_CUDA_H

#include <vector>

std::vector<float> NaiveGemmCUDA(const std::vector<float>& mxA,
    const std::vector<float>& mxB,
    int src);

#endif // __NAIVE_GEMM_CUDA_H