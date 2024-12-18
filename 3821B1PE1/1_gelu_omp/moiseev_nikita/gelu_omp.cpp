// Copyright (c) 2024 Moiseev Nikita
#include "gelu_omp.h"

#include <omp.h>
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }
    const float gelu_coefficient = 0.044715f;
    size_t input_size = input.size();
    std::vector<float> output(input_size);

#pragma omp parallel for
    for (size_t index = 0; index < input_size; ++index) {
        float value = input[index];
        output[index] = 0.5f * value * (1.0f + tanh(sqrt(2.0f / M_PI)
            * (value + gelu_coefficient * value * value * value)));
    }

    return output;
}