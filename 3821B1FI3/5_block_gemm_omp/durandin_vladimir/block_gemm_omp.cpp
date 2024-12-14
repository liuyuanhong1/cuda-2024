// Copyright (c) 2024 Durandin Vladimir

#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b, int n) {
  std::vector<float> result(n * n, 0);

  int blockSize = 16;

#pragma omp parallel for collapse(2)
  for (int blockRow = 0; blockRow < n; blockRow += blockSize) {
    for (int blockCol = 0; blockCol < n; blockCol += blockSize) {
      for (int blockK = 0; blockK < n; blockK += blockSize) {
        for (int row = blockRow; row < std::min(blockRow + blockSize, n);
             ++row) {
          for (int col = blockCol; col < std::min(blockCol + blockSize, n);
               ++col) {
            float partialSum = 0.0f;
            for (int k = blockK; k < std::min(blockK + blockSize, n); ++k) {
              partialSum += a[row * n + k] * b[k * n + col];
            }
#pragma omp atomic
            result[row * n + col] += partialSum;
          }
        }
      }
    }
  }

  return result;
}
