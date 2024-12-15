// Copyright (c) 2024 Khodyrev Fedor
#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float> &var1, const std::vector<float> &var2, int size) {
  auto countElem = size * size;
  if (var1.size() != countElem || var2.size() != countElem) return {};

  std::vector<float> var3(countElem, 0.0f);
  const auto blockSize = 8;
  auto numBlocks = size / blockSize;

#pragma omp parallel for shared(var1, var2, var3)
  for (int i = 0; i < numBlocks; ++i)
    for (int j = 0; j < numBlocks; ++j)
      for (int block = 0; block < numBlocks; ++block)
        for (int m = i * blockSize; m < (i + 1) * blockSize; ++m)
          for (int n = j * blockSize; n < (j + 1) * blockSize; ++n)
            for (int k = block * blockSize; k < (block + 1) * blockSize; ++k)
              var3[m * size + n] += var1[m * size + k] * var2[k * size + n];

  return var3;
}