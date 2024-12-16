// Copyright (c) 2024 Kostanyan Arsen
#include "gelu_omp.h"
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float> &input) {
  if (input.empty()) return {};

  constexpr float constOne = 1.595769122f;
  constexpr float constTwo = constOne * 0.044715f;

  auto size = input.size();
  std::vector<float> output(size);

#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    float value = input[i];
    float tmp = value * (constOne + value * value * constTwo);
    output[i] = value - value / (1.0f + std::exp(tmp));
  }

  return output;
}
