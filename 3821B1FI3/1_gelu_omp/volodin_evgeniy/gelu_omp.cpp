// Copyright (c) 2024 Volodin Evgeniy
#include "gelu_omp.h"
#include <omp.h>
#include <cmath>
#include <stdexcept>

std::vector<float> GeluOMP(const std::vector<float> &input) {
    if (input.empty()) {
      throw std::invalid_argument("Input vector is empty!");
    }

    std::vector<float> output(input.size());

    const float sqrt_2pi = std::sqrt(2.0f / M_PI);
    const float coeff_cubic = 0.044715f;

    #pragma omp parallel for
    for (std::size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + std::tanh(sqrt_2pi * (x + coeff_cubic * x * x * x)));
    }

    return output;
}