// Copyright (c) 2024 Podyachikh Mikhail
#include "gelu_omp.h"

#include <omp.h>
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const float gelu_coefficient = 0.044715f;
    const float sqrt_two_over_pi = sqrt(2.0f / M_PI);

    size_t vector_size = input.size();
    std::vector<float> output(vector_size);

#pragma omp parallel for
    for (size_t index = 0; index < vector_size; ++index) {
        float value = input[index];
        output[index] = 0.5f * value *
            (1.0f + tanh(sqrt_two_over_pi * (value + gelu_coefficient * value * value * value)));
    }

    return output;
}
