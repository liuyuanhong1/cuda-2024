// Copyright (c) 2024 Afanasyev Aleksey

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
    float valueSquared = value * value;
    float coeffTerm = geluCoeff1 + valueSquared * geluCoeff2;
    float expTerm = std::exp(value * coeffTerm);
    float denominator = 1.0f + expTerm;
    float scaledValue = 1 - 1 / denominator;
    output[i] = value * scaledValue;
  }

  return output;
}
