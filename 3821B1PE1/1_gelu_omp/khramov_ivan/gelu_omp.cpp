// Copyright (c) 2024 Khramov Ivan
#include "gelu_omp.h"

#include <omp.h>
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    if (input.empty()) return {};

    const float geluConst = 0.044715f;

    auto inputSize = input.size();
    std::vector<float> output(inputSize);

#pragma omp parallel for
  for (size_t i = 0; i < inputSize; ++i) {
    float x = input[i];
    output[i] = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI)
     * (x + geluConst * x * x * x)));
  }

  return output;
}