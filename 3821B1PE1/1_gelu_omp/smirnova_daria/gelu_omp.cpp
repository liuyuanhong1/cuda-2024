// Copyright (c) 2024 Smirnova Daria
#include "gelu_omp.h"

#include <omp.h>
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    if (input.empty()) return {};

    const float geluConst = 0.7978845608f;
    const float geluConst2 = 0.044715f;

    std::vector<float> result(input.size());

#pragma omp parallel for
  for (size_t i = 0; i < input.size(); ++i) {
    float x = input[i];

    result[i] = 0.5f * x * (1.0f + tanh(geluConst * (x + geluConst2 * x * x * x)));
  }
  return result;
}