#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b, int n) {
  std::vector<float> c(n * n, 0.0f);
  size_t i = 0;
  size_t j = 0;
  size_t k = 0;

#pragma omp parallel for shared(matrA, matrB, matrC) private(m, n, k) \
    collapse(3)
  for (i = 0; i < n; ++i)
    for (j = 0; j < n; ++j)
      for (k = 0; k < n; ++k) c[i * n + j] += a[i * n + k] * b[n * k + j];

  return c;
}
