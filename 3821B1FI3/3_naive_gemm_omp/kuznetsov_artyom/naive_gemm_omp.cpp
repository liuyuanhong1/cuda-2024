// Copyright (c) 2024 Kuznetsov-Artyom
#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b, int size) {
  std::size_t countElem = size * size;
  if (a.size() != countElem || b.size() != countElem) return {};

  std::vector<float> c(countElem, 0.0f);
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;

#pragma omp parallel for shared(a, b, c) private(m, n, k) collapse(3)
  for (m = 0; m < size; ++m)
    for (n = 0; n < size; ++n)
      for (k = 0; k < size; ++k)
        c[m * size + n] += a[m * size + k] * b[size * k + n];

  return c;
}
