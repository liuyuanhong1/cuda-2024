// Copyright (c) 2024 Nedelin Dmitry

#ifndef __NAIVE_GEMM_OMP_H
#define __NAIVE_GEMM_OMP_H

#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& matrix_a,
    const std::vector<float>& matrix_b,
    int dimension);

#endif // __NAIVE_GEMM_OMP_H
