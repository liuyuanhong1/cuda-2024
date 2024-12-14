// Copyright (c) 2024 Saratova Marina
#include "gelu_omp.h"

#include <cmath>

std::vector<float> GeluOMP(const std::vector<float> &input) {
  if (input.empty()) return {};
  const float constOne = 0.79788456f;
  const float constTwo = 0.044715f;
  const auto size = input.size();
  std::vector<float> output(size);

#pragma omp parallel for
  for (size_t i = 0; i < size; i++) {
    float value = input[i];
    float tmp = tanh(constOne * (value + constTwo * value * value * value));
    output[i] = 0.5f * value * (1.0f + tmp);
  }

  return output;
}
