// Copyright (c) 2024 Kulagin Aleksandr
#define _USE_MATH_DEFINES
#include "fft_cufft.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>

static void test_time() {
  const std::vector<float>::size_type sz = 131072;
  const int batch = 256;
  std::vector<float> input(sz);
  for (std::vector<float>::size_type i = 0; i < sz; i++) {
    input[i] = (float)((i + 1) % 1000) / M_PIf;
  }
  std::vector<float> input_pre(input);
  input_pre.resize(sz / 8);
  FffCUFFT(input_pre, batch);

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<float> out = FffCUFFT(input, batch);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "ELAPSED TIME cuFFT " << duration.count() << '\n';
}

int main() {
  test_time();
  return 0;
}
