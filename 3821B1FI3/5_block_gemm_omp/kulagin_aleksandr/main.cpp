// Copyright (c) 2024 Kulagin Aleksandr
#include "block_gemm_omp.h"
#include <vector>
#define _USE_MATH_DEFINES
#include <omp.h>
#include <cmath>
#include <limits>
#include <iostream>
#include <cassert>
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
  int n = 32;
  int sz = n * n;
  std::vector<float> a(sz);
  std::vector<float> b(sz);
  for (int i = 0; i < sz; i++) {
    a[i] = (float)(i + 1) / M_PIf; // meh
    b[i] = (float)(i - 1) / M_PIf;
  }
  std::vector<float> out1 = matrix_mul(a, b, n), out2 = BlockGemmOMP(a, b, n);
  for (int i = 0; i < sz; i++) {
    if (std::abs(out1[i] - out2[i]) > std::numeric_limits<float>::epsilon()) {
      std::cout << "BAD VALUE AT " << i << ' ' << out1[i] << ' ' << out2[i] << " WITH DIFFERENCE " << (out1[i] - out2[i]) << '\n';
      ret = 1;
    }
  }
  return ret;
}

static void test_time() {
  //omp_set_num_threads(4);
  int n = 32; // 1024
  int sz = n * n;
  std::vector<float> a(sz);
  std::vector<float> b(sz);
  for (int i = 0; i < sz; i++) {
    a[i] = (float)(i + 1) / M_PIf; // meh
    b[i] = (float)(i - 1) / M_PIf;
  }
  BlockGemmOMP(b, a, n);

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<float> out = BlockGemmOMP(a, b, n);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "ELAPSED TIME OMP " << duration.count() << '\n';
  //omp_set_num_threads(omp_get_max_threads());
}

int main() {
  int ret = test_correctness();
  assert(ret == 0);
  test_time();
  return ret;
}
