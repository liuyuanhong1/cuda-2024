// Copyright (c) 2024 Kulagin Aleksandr
#define _USE_MATH_DEFINES
#include "naive_gemm_omp.h"
#include "test_matrices.h"
#include <omp.h>
#include <cmath>
#include <limits>
#include <iostream>
#include <cassert>
#include <vector>
#include <chrono>

static int test_correctness() {
  int ret = 0;
  const int n = TEST_ARRAY_N;
  const int sz = n * n;
  const std::vector<float> a = { A_TEST_ARRAY };
  const std::vector<float> b = { B_TEST_ARRAY };
  const std::vector<float> out_real = { C_TEST_ARRAY };
  assert((int)a.size() == sz);
  assert((int)b.size() == sz);
  assert((int)out_real.size() == sz);
  const std::vector<float> out = NaiveGemmOMP(a, b, n);
  assert(out_real.size() == out.size());
  const float max_diff = 2.e-05; // I hate python sometimes
  for (int i = 0; i < sz; i++) {
    if (std::abs(out_real[i] - out[i]) > max_diff) {
      std::cout << "BAD VALUE AT " << '[' << i % sz << ',' << i / sz << ']' << ' ' << out_real[i] << ' ' << out[i] << " WITH DIFFERENCE " << std::abs(out_real[i] - out[i]) << '\n';
      ret = 1;
    }
  }
  return ret;
}

static void test_time() {
  //omp_set_num_threads(4);
  const int sz = 1024;
  const int n = std::sqrt(sz);
  assert(sz == n * n);
  std::vector<float> a(sz);
  std::vector<float> b(sz);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      a[i * n + j] = (float)(i + j) / M_PIf;
      b[j * n + i] = (float)(i + j) / M_PIf;
    }
  }
  {
    // warmup
    int test_sz = std::sqrt(sz / 8);
    test_sz *= test_sz;
    const std::vector<float> a(test_sz);
    const std::vector<float> b(test_sz);
    const std::vector<float> out = NaiveGemmOMP(a, b, std::sqrt(test_sz));
  }
  const auto start = std::chrono::high_resolution_clock::now();
  const std::vector<float> out = NaiveGemmOMP(a, b, n);
  const auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> duration = end - start;
  std::cout << "ELAPSED TIME OMP " << duration.count() << '\n';
  //omp_set_num_threads(omp_get_max_threads());
}

int main() {
  const int ret = test_correctness();
  assert(ret == 0);
  test_time();
  return ret;
}
