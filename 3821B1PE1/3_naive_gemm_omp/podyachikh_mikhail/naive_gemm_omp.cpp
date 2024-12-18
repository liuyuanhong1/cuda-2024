// Copyright (c) 2024 Podyachikh Mikhail
#include "naive_gemm_omp.h"

#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b,
                                int n) {
  std::vector<float> c(n * n);

#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < n; k++) {
      for (int j = 0; j < n; j++) {
        c[i * n + j] += a[i * n + k] * b[k * n + j];
      }
    }
  }
  return c;
}
