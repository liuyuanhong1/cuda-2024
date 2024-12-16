// Copyright (c) 2024 Nogin Denis
#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b, int size) {
  auto countElem = size * size;
  if (a.size() != countElem || b.size() != countElem) return {};

  std::vector<float> c(countElem, 0.0f);
  constexpr auto blockSize = 8;
  auto numBlocks = size / blockSize;

#pragma omp parallel for shared(a, b, c)
  for (int blockRow = 0; blockRow < numBlocks; ++blockRow) {
    for (int blockCol = 0; blockCol < numBlocks; ++blockCol) {
      for (int block = 0; block < numBlocks; ++block) {
        for (int row = blockRow * blockSize; row < (blockRow + 1) * blockSize; ++row) {
          for (int col = blockCol * blockSize; col < (blockCol + 1) * blockSize; ++col) {
            for (int k = block * blockSize; k < (block + 1) * blockSize; ++k) {
              c[row * size + col] += a[row * size + k] * b[k * size + col];
            }
          }
        }
      }
    }
  }

  return c;
}