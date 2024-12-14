// Copyright (c) 2024 Kazantsev Evgeny
#include "gelu_omp.h"
#include <cmath>
#include <vector>

constexpr float scalar1 = 0.044715f;
constexpr float scalar2 = 0.5f;

std::vector<float> GeluOMP(const std::vector<float>& input) {

  if (input.empty()) {
    return {};
  }

  std::vector<float> output(input.size());

  #pragma omp parallel for
  for (size_t i = 0; i < input.size(); ++i) {

    float x = input[i];
    float tanh_arg = std::sqrt(2 / M_PI) 
                    * (x + scalar1 * std::pow(x, 3));

    output[i] = scalar2 * x * (1 + std::tanh(tanh_arg));
  }
  return output;
}