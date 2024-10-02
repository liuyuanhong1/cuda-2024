//Copyright Smirnov Pavel
#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
  std::vector<float> output(input.size());

  float sqrt_2pi = std::sqrt(2.0f / M_PI);

  #pragma omp parallel for
  for (size_t i = 0; i < input.size(); ++i) {
    float angle = sqrt_2pi * (input[i] + 0.044715f * std::pow(input[i], 3));
    output[i] = 0.5f * input[i] * (1.0f + std::tanh(angle));
  }

  return output;
}

