// Copyright (c) 2024 Nedelin Dmitry

#ifndef __GEMM_CUBLAS_H
#define __GEMM_CUBLAS_H

#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& matrixA,
                              const std::vector<float>& matrixB,
                              int dimension);

#endif // __GEMM_CUBLAS_H
