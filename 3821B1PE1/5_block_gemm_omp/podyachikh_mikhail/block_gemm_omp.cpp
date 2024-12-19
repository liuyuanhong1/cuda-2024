// Copyright (c) 2024 Podyachikh Mikhail
#include "block_gemm_omp.h"

#include <cassert>
#include <algorithm>


std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b,
                                int n) {
  if (n == 0) return {};

  std::vector<float> c(n * n);
  const int block_size = std::min(n, 16);
  assert(n % block_size == 0);

#pragma omp parallel for
  for (int ii = 0; ii < n; ii += block_size) {
    for (int kk = 0; kk < n; kk += block_size) {
      for (int jj = 0; jj < n; jj += block_size) {
        for (int i = 0; i < block_size; i++) {
          for (int k = 0; k < block_size; k++) {
            for (int j = 0; j < block_size; j++) {
              c[(ii + i) * n + (jj + j)] +=
                  a[(ii + i) * n + (kk + k)] * b[(kk + k) * n + (jj + j)];
            }
          }
        }
      }
    }
  }

  return c;
}
