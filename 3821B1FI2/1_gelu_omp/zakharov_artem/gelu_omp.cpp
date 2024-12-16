// Copyright (c) 2024 Zakharov-Artem
#include "gelu_omp.h"

#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> result(n);
    constexpr float SQRT_TWO_OVER_PI = 0.797885;

    #pragma omp parallel for default(none) shared(n, input, result, SQRT_TWO_OVER_PI)
    for (size_t i = 0; i < n ; i++) {
        float x = input[i];
        result[i] = 0.5f * x * (1 + std::tanh(SQRT_TWO_OVER_PI * x * (1 + 0.044715f * x * x)));
    }
    return result;
}
