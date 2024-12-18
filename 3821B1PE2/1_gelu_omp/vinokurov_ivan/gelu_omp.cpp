//  Copyright (c) 2024 Vinokurov Ivan
#include "gelu_omp.h"
#include <cmath>


std::vector<float> GeluOMP(const std::vector<float>& input) {
    if (input.empty())
        return {};

    float ratio1 = 1.595769122f;
    float ratio2 = 0.071354816f;

    auto inputSize = input.size();
    std::vector<float> output(inputSize);

#pragma omp parallel for
    for (size_t i = 0; i < inputSize; ++i) {
        float value = input[i];
        output[i] = value * (1 - 1 / (1.0f + std::exp(value * (ratio1 + value * value * ratio2))));
    }

    return output;
}