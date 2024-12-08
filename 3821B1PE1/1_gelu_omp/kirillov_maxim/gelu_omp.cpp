// Copyright (c) 2024 Kirillov Maxim
#include "gelu_omp.h"
#include <math.h>
#include <omp.h>

const float geluParameter = 0.044715f;
const float piParameter = sqrt(2.0f / M_PI);

std::vector<float> GeluOMP(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }
    const size_t input_size = input.size();
    std::vector<float> output(input_size);

#pragma omp parallel for
    for (size_t i = 0; i < input.size(); i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + tanh(piParameter * (x + geluParameter * x * x * x)));
    }
    return output;
}