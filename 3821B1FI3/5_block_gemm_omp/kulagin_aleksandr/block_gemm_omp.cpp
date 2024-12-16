// Copyright (c) 2024 Kulagin Aleksandr
#include "block_gemm_omp.h"
#include <omp.h>
#include <cassert>

constexpr int block_size = 32;

std::vector<float> BlockGemmOMP(const std::vector<float> &a, const std::vector<float> &b, int n) {
  if (n == 0) {
    return std::vector<float>();
  }
  assert(a.size() == n * n);
  assert(b.size() == n * n);
  std::vector<float> res(n * n, 0.0f);
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; i += block_size) {
    for (int j = 0; j < n; j += block_size) {
      for (int k = 0; k < n; k += block_size) {
        for (int it = i; it < i + block_size && it < n; it++) {
          for (int jt = j; jt < j + block_size && jt < n; jt++) {
            float tmp = 0.0f;
            for (int kt = k; kt < k + block_size && kt < n; kt++) {
              tmp += a[it * n + kt] * b[kt * n + jt];
            }
            res[it * n + jt] += tmp;
          }
        }
      }
    }
  }
  return res;
}
