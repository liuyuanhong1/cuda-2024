// Copyright (c) 2024 Morgachev Stepan
#include "gelu_omp.h"
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    if (input.empty()) return {};

    constexpr float sqrtTwoOverPi = std::sqrt(2.0f / M_PI);
    constexpr float coeff = 0.044715f;

    auto size = input.size();
    std::vector<float> result(size);

    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        float currentValue = input[i];
        float cubicValue = currentValue * currentValue * currentValue;
        float tanhInput = sqrtTwoOverPi * (currentValue + coeff * cubicValue);
        result[i] = 0.5f * currentValue * (1.0f + std::tanh(tanhInput));
    }

    return result;
}
