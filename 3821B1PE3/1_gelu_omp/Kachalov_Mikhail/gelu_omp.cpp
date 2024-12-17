// Copyright 2024 Kachalov Mikhail

#include <omp.h>
#include <vector>
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float> &input)
{
    const float sqrt_pi = std::sqrt(2.0 / M_PI);
    const float c = 0.044715;

    size_t size = input.size();
    std::vector<float> result(size);

#pragma omp parallel for
    for (size_t i = 0; i < size; ++i)
    {
        float x = input[i];
        float cube_term = c * x * x * x;
        float tanh_arg = sqrt_pi * (x + cube_term);
        float tanh_value = std::tanh(tanh_arg);
        result[i] = 0.5f * x * (1.0f + tanh_value);
    }

    return result;
}