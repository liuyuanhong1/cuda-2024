// Copyright (c) 2024 Durandin Vladimir

#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b, int size) {

  auto totalElements = size * size;
  if (a.size() != totalElements || b.size() != totalElements)
    return {};

  std::vector<float> resultMatrix(totalElements, 0.0f);

#pragma omp parallel for
  for (int row = 0; row < size; ++row) {
    for (int inner = 0; inner < size; ++inner) {
      float tempValue = a[row * size + inner];
      for (int col = 0; col < size; ++col) {
        resultMatrix[row * size + col] += tempValue * b[inner * size + col];
      }
    }
  }

  return resultMatrix;
}
