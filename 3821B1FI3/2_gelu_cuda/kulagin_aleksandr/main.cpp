// Copyright (c) 2024 Kulagin Aleksandr
#define _USE_MATH_DEFINES
#include "gelu_cuda.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <cassert>
#include <chrono>

static std::vector<float> GeluSEQ(const std::vector<float>& input) {
  if (input.empty()) {
    return std::vector<float>();
  }
  const float precalc_c_1 = std::sqrt(2.0f / M_PIf);
  const std::vector<float>::size_type sz = input.size();
  std::vector<float> res(sz);
  for (std::vector<float>::size_type i = 0; i < sz; i++) {
    const float& x = input[i];
    res[i] = 0.5f * x * (1.0f + std::tanh(precalc_c_1 * ( x + 0.044715f * (x * x * x) )));
  }
  return res;
}

static int test_correctness() {
  int ret = 0;
  const std::vector<float>::size_type sz = 1024;
  std::vector<float> input(sz);
  for (std::vector<float>::size_type i = 0; i < sz; i++) {
    input[i] = (float)(i + 1) / M_PIf; // meh
  }
  std::vector<float> out1 = GeluSEQ(input), out2 = GeluCUDA(input);
  for (std::vector<float>::size_type i = 0; i < sz; i++) {
    if (std::abs(out1[i] - out2[i]) > std::numeric_limits<float>::epsilon()) {
      std::cout << "BAD VALUE AT " << i << ' ' << out1[i] << ' ' << out2[i] << " WITH DIFFERENCE " << (out1[i] - out2[i]) << '\n';
      ret = 1;
    }
  }
  return ret;
}

static void test_time() {
  const std::vector<float>::size_type sz = 134217728;
  std::vector<float> input(sz);
  for (std::vector<float>::size_type i = 0; i < sz; i++) {
    input[i] = (float)(i + 1) / M_PIf;
  }
  std::vector<float> input_pre(input);
  input_pre.resize(sz / 8);
  GeluCUDA(input_pre);

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<float> out = GeluCUDA(input);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "ELAPSED TIME CUDA " << duration.count() << '\n';

  start = std::chrono::high_resolution_clock::now();
  std::vector<float> out2 = GeluSEQ(input);
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "ELAPSED TIME SEQ " << duration.count() << '\n';
}

int main() {
  int ret = test_correctness();
  assert(ret == 0);
  test_time();
  return ret;
}
