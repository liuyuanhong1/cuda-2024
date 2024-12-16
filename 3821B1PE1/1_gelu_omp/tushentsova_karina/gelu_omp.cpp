// Copyright (c) 2024 Tushentsova Karina
#include "gelu_omp.h"

#include <omp.h>
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    const float coeff = 0.044715f;

    size_t size = input.size();
    std::vector<float> output(size);

#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        float value = input[i];
        output[i] = 0.5f * value * (1.0f + tanh(sqrt(2.0f / M_PI)
            * (value + coeff * value * value * value)));
    }

    return output;
}