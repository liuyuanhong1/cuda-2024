// Copyright (c) 2024 Kulikov Artem
#include "gelu_omp.h"

#include <cmath>

std::vector<float> GeluOMP(const std::vector<float> &input) {
  if (input.empty()) return {};

  const size_t sz = input.size();
  std::vector<float> output(sz);

  const float isqrt2 = 1.0f / sqrtf(2.0f);

#pragma omp parallel for
  for (size_t i = 0; i < sz; i++) {
    output[i] = 0.5f * input[i] * (1.0f + erff(input[i] * isqrt2));
  }

  return output;
}
