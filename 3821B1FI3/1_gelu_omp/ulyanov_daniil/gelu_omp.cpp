// Copyright (c) 2024 Ulyanov Daniil

#include "gelu_omp.h"

#include <cmath>

std::vector<float> GeluOMP(const std::vector<float> &input) {
  if (input.empty())
    return std::vector<float>{};

  auto fast_tanh = [](const float &x) -> float {
    return 2.0f / (1.0f + std::exp(-2.0f * x)) - 1.0f;
  };

  constexpr const float PI{3.14159265358979f};
  constexpr const float sqrt_2_div_pi{0.79788456080286f};

  size_t input_size = input.size();
  std::vector<float> gelu_result(input_size);

  float x{};

#pragma omp parallel for simd private(x)
  for (size_t i = 0; i < input_size; ++i) {
    x = input[i];
    gelu_result[i] =
        0.5f * x *
        (1.0f + fast_tanh(sqrt_2_div_pi * (x + 0.044715f * x * x * x)));
  }
  return gelu_result;
}
