// Copyright (c) 2024 Prokofev Kirill
#include "block_gemm_omp.h"
#include <iostream>
#include <chrono>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b, int n) {
  std::vector<float> result(n * n, 0.0f);
  int block_size = 32;
  #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int k = 0; k < n; k += block_size) {
                for (int ii = i; ii < i + block_size && ii < n; ++ii) {
                    for (int jj = j; jj < j + block_size && jj < n; ++jj) {
                        float sum = result[ii * n + jj];
                        int kk;
                        for (kk = k; kk <= k + block_size - 4 && kk < n; kk += 4) {
                            sum += a[ii * n + kk] * b[kk * n + jj];
                            sum += a[ii * n + kk + 1] * b[(kk + 1) * n + jj];
                            sum += a[ii * n + kk + 2] * b[(kk + 2) * n + jj];
                            sum += a[ii * n + kk + 3] * b[(kk + 3) * n + jj];
                        }
                        for (; kk < k + block_size && kk < n; ++kk) {
                            sum += a[ii * n + kk] * b[kk * n + jj];
                        }
                        
                        result[ii * n + jj] = sum;
                    }
                }
            }
        }
    }
    return result;
}

