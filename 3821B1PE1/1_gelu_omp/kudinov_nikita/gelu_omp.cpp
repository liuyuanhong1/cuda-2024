// Copyright (c) 2024 Kudinov Nikita

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
    output[i] = input[i] * (
      1 - 1 / (1.0f + std::exp(input[i] * (geluCoeff1 + input[i] * input[i] * geluCoeff2)))
    );
  }

  return output;
}
