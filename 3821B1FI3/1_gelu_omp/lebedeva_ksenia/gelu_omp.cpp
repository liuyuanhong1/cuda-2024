// Copyright (c) 2024 Lebedeva Ksenia
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
    float val = input[i];
    float tmp = val * (constOne + val * val * constTwo);
    output[i] = val - val / (1.0f + std::exp(tmp));
  }

  return output;
}
