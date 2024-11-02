// Copyright (c) 2024 Korablev Nikita
#include "gelu_omp.h"
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    if (input.empty()) return {};
    std::vector<float> output(input.size());

    const float a = 0.5f;
    const float b = 0.044715f;
    const float sqr = std::sqrt(2.0f / M_PI);

#pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float tan = std::tanh(sqr * (x + b * x * x * x));
        output[i] = a * x * (1.0f + tan);
    }

    return output;
}
