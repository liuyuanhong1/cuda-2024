// Copyright (c) 2024 Vinichuk Timofey

#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float> &input) {
  constexpr float geluCoef1 = 0.044715f;
  constexpr float geluCoef2 = 0.7978845608f;

  auto size = input.size();
  std::vector<float> output(size);

#pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    float x = input[i];
    output[i] = 0.5f * x * (1.0f + tanhf(geluCoef2 * x * (1.0f + geluCoef2 * x * x)));
  }

  return output;
}
