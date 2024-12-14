// Copyright (c) 2024 Vanushkin Dmitry
#include "gelu_omp.h"
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
  constexpr float magicConst = 0.044715f;
  // Evaluated sqrt(PI/2)
  constexpr float k = 0.7978845608f;

  auto size = input.size();
  std::vector<float> output(size);

#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    auto x = input[i];
    auto gelu = 0.5f * x * (1 + tanh(k * (x + magicConst * std::pow(x, 3.0))));
    output[i] = gelu;
  }

  return output;
}
