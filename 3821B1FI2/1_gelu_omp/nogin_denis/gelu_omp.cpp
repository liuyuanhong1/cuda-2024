// Copyright (c) 2024 Nogin Denis
#include "gelu_omp.h"

#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    if (input.empty()) return {};
    
    auto size = input.size();
    std::vector<float> result(size);

    const float constOne = 1.595769122f;
    const float constTwo = constOne * 0.044715f;

#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        float x = input[i];
        float temp = x * (constOne + x * x * constTwo);
        result[i] = x - x / (1.0f + std::exp(temp));
    }

    return result;
}