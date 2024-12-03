// Copyright (c) 2024 Soloninko Andrey
#include "gelu_omp.h"
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float> &input) {
  if (input.empty()) return {};

  const float constOne = 1.595769122f;
  const float constTwo = 1.595769122f * 0.044715f;

  auto size = input.size();
  std::vector<float> output(size);

#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    float val = input[i];
    float temp = val * (constOne + val * val * constTwo);
    output[i] = val - val / (1.0f + std::exp(temp));
  }

  return output;
}
