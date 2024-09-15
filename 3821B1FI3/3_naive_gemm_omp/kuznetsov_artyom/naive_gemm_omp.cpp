#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b, int size) {
  std::vector<float> c(size * size, 0.0f);
  size_t m = 0;
  size_t n = 0;
  size_t k = 0;

#pragma omp parallel for shared(a, b, c) private(m, n, k) collapse(3)
  for (m = 0; m < size; ++m)
    for (n = 0; n < size; ++n)
      for (k = 0; k < size; ++k)
        c[m * size + n] += a[m * size + k] * b[size * k + n];

  return c;
}
