// Copyright (c) 2024 Sokolova Daria
#include "gelu_omp.h"
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    if (input.empty()) return {};

    constexpr float factor = std::sqrt(2.0f / M_PI);
    constexpr float cubicCoeff = 0.044715f;

    size_t size = input.size();
    std::vector<float> output(size);

    #pragma omp parallel for
    for (size_t index = 0; index < size; ++index) {
      float curr = input[index];
      output[index] = 0.5f * curr * (1.0f + std::tanh(factor * (curr + cubicCoeff * curr * curr * curr)));
    }

    return output;
}
