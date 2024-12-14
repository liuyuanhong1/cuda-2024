#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                   const std::vector<float>& b, 
                                   int n) {
  std::vector<float> result(n * n, 0.0f);
  int blockSize = 32;

  #pragma omp parallel for collapse(2) schedule(dynamic)
  for (int i = 0; i < n; i += blockSize) {
    for (int j = 0; j < n; j += blockSize) {
      for (int k = 0; k < n; k += blockSize) {
        for (int ii = i; ii < std::min(i + blockSize, n); ++ii) {
          for (int kk = k; kk < std::min(k + blockSize, n); ++kk) {
            float aik = a[ii * n + kk];

            #pragma omp simd
            for (int jj = j; jj < std::min(j + blockSize, n); ++jj) {
              result[ii * n + jj] += aik * b[kk * n + jj];
            }
          }
        }
      }
    }
  }

  return result;
}
