// Copyright (c) 2024 Kulikov Artem
#include "gelu_omp.h"

#include <cmath>

std::vector<float> GeluOMP(const std::vector<float> &input) {
  if (input.empty()) return {};

  const size_t sz = input.size();
  std::vector<float> output(sz);

  const float c = 0.044715f;
  const float dsqrt2ipi = 1.59577f;  // 2 * sqrt(2 / PI)

#pragma omp parallel for
  for (size_t i = 0; i < sz; i++) {
    output[i] = input[i] / (1.0f + std::exp(-dsqrt2ipi * (input[i] + c * input[i] * input[i] * input[i])));
  }

  return output;
}
