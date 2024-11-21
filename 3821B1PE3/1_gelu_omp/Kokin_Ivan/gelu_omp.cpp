//Copyright 2024 Kokin Ivan

#define _USE_MATH_DEFINES
#include "gelu_omp.h"
#include <cmath>
#include <vector>


constexpr float twoPi = std::sqrt(2.0f / static_cast<float>(M_PI));

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());

#pragma omp parallel for
    for (std::size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float poly = x + 0.044715f * x * x * x;
        float tanh_val = std::tanh(twoPi * poly);
        output[i] = 0.5f * x * (1.0f + tanh_val);
    }

    return output;
}
