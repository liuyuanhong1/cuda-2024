// Copyright (c) 2024 Kulagin Aleksandr
#define _USE_MATH_DEFINES
#include "block_gemm_cuda.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <cassert>
#include <vector>
#include <chrono>

static std::vector<float> matrix_mul(const std::vector<float> &a, const std::vector<float> &b, int n) {
  if (n == 0) {
    return std::vector<float>();
  }
  assert(a.size() == n * n);
  assert(b.size() == n * n);
  std::vector<float> res(n * n, 0.0f);
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

static int test_correctness() {
  int ret = 0;
  int n = 16;
  int sz = n * n;
  std::vector<float> a(sz);
  std::vector<float> b(sz);
  for (int i = 0; i < sz; i++) {
    a[i] = (float)((i + 1) % 100) / M_PIf; // meh
    b[i] = (float)((i - 1) % 100) / M_PIf;
  }
  std::vector<float> out1 = matrix_mul(a, b, n), out2 = BlockGemmCUDA(a, b, n);
  for (int i = 0; i < sz; i++) {
    if (std::abs(out1[i] - out2[i]) > 0.001f) { // weird
      std::cout << "BAD VALUE AT " << i << ' ' << out1[i] << ' ' << out2[i] << " WITH DIFFERENCE " << (out1[i] - out2[i]) << '\n';
      ret = 1;
    }
  }
  return ret;
}

static void test_time() {
  int n = 64; // 4096
  int sz = n * n;
  std::vector<float> a(sz);
  std::vector<float> b(sz);
  for (int i = 0; i < sz; i++) {
    a[i] = (float)((i + 1) % 100) / M_PIf; // meh
    b[i] = (float)((i - 1) % 100) / M_PIf;
  }
  BlockGemmCUDA(b, a, n);

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<float> out = BlockGemmCUDA(a, b, n);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "ELAPSED TIME CUDA " << duration.count() << '\n';
}

int main() {
  int ret = test_correctness();
  assert(ret == 0);
  test_time();
  return ret;
}
