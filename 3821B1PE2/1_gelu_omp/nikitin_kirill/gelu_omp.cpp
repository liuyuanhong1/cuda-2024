// Copyright (c) 2024 Nikitin Kirill

#include "gelu_omp.h"
#include <cmath>


std::vector<float> GeluOMP(const std::vector<float> &input) {
  if (input.empty()) return {};

  constexpr float geluCoeff1 = 1.595769122f;
  constexpr float geluCoeff2 = 0.071354816f;    

  auto size = input.size();
  std::vector<float> output(size);

#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    float value = input[i];
    output[i] = value * (1 - 1 / (1.0f + std::exp(value * (geluCoeff1 + value * value * geluCoeff2))));
  }

  return output;
}
