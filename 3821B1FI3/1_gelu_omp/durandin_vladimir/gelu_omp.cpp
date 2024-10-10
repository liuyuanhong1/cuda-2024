// Copyright (c) 2024 Durandin Vladimir

#include "gelu_omp.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>

// ----

// It is better to do it this way for vectorization, instead of a lambda
// expression inside the GELU function

// #if defined(_MSC_VER)
//   #define INLINE_ATTR __forceinline
// #elif defined (__GNUC__) || defined(__clang__)
//   #define INLINE_ATTR [[gnu::always_inline]]
// #else
//   #define INLINE_ATTR inline
// #endif

// Approximation through exponentials (approximate sigmoid)
// INLINE_ATTR float fast_tanh(const float &x) {
//   return 2.0f / (1.0f + std::exp(-2.0f * x)) - 1.0f;
// }

// -----

std::vector<float> GeluOMP(const std::vector<float> &input) {
  if (input.empty())
    return std::vector<float>{};

  // Approximation through exponentials (approximate sigmoid)
  auto fast_tanh = [](const float &x) -> float {
    return 2.0f / (1.0f + std::exp(-2.0f * x)) - 1.0f;
  };

  constexpr const float PI{3.14159265358979f};
  constexpr const float sqrt_2_div_pi{0.79788456080286f}; // sqrt(2.0 / PI)

  size_t input_size = input.size();
  std::vector<float> gelu_result(input_size);

  double x{};

#pragma omp parallel for simd private(x)
  for (size_t i = 0; i < input_size; ++i) {
    x = input[i];
    gelu_result[i] =
        0.5f * x *
        (1.0f + fast_tanh(sqrt_2_div_pi * (x + 0.044715f * x * x * x)));
  }
  return gelu_result;
}

int main() {

  std::vector<float> test(134217728);
  for (auto &val : test) {
    val = static_cast<float>(std::rand()) /
          RAND_MAX; // Случайное число в диапазоне [0, 1)
  }
  auto start = std::chrono::high_resolution_clock::now();
  // for (int i = 0; i < 1000; ++i) {
    GeluOMP(test);
  // }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  // std::chrono::duration<double, std::milli> duration = end - start;

  std::cout << "Time of execution: " << duration.count() << " milli "
            << std::endl;
  return 0;
}
