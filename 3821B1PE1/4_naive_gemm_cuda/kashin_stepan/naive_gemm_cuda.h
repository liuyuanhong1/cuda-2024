// Copyright (c) 2024 Kashin Stepan

#ifndef __MATRIX_MULTIPLICATION_CUDA_H
#define __MATRIX_MULTIPLICATION_CUDA_H

#include <vector>

std::vector<float> PerformMatrixMultiplicationCUDA(const std::vector<float>& matrixA,
                                                  const std::vector<float>& matrixB, int size);

std::vector<float> PerformMatrixMultiplicationCUDA_v2(const std::vector<float>& matrixA,
                                                  const std::vector<float>& matrixB, int size);

#endif  // __MATRIX_MULTIPLICATION_CUDA_H