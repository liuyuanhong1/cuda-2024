// Copyright (c) 2024, Kriseev Mikhail

#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b, int n) {
  std::vector<float> output(n * n, 0.0f);
  float *output_data = output.data();
  const float *a_data = a.data();
  const float *b_data = b.data();

  int blockSize = 16;
  int numBlocks = n / blockSize;

  if (numBlocks == 0) {
    blockSize = n;
    numBlocks = 1;
  }

#pragma omp parallel for
  for (int i = 0; i < numBlocks; ++i) {
    for (int j = 0; j < numBlocks; ++j) {
      for (int k = 0; k < numBlocks; ++k) {
        //
        #pragma omp parallel for
        for (int l = 0; l < blockSize; ++l) {
          for (int m = 0; m < blockSize; ++m) {
            float res = 0.0f;
            for (int k1 = 0; k1 < blockSize; ++k1) {
              res += a[(i * blockSize + l) * n + k * blockSize + k1] *
                     b[(k * blockSize + k1) * n + j * blockSize + m];
            }
            output_data[(i * blockSize + l) * n + j * blockSize + m] += res;
          }
        }
      }
    }
  }

  return output;
}