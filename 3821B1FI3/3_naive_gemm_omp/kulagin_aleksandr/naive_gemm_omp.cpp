// Copyright (c) 2024 Kulagin Aleksandr
#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n) {
  const int res_size = n * n;
  if ((int)a.size() != res_size || (int)b.size() != res_size) {
    return std::vector<float>();
  }
  std::vector<float> res(res_size, 0.0f);
#pragma omp parallel for schedule(static) collapse(2)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float& res_ij = res[i * n + j];
      for (int k = 0; k < n; k++) {
        res_ij += a[i * n + k] * b[k * n + j];
      }
    }
  }
  return res;
}
