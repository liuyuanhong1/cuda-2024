#include "gelu_omp.h"

#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    
    auto size_input = input.size();

    if (size_input == 0) {
        return {};
    }

     constexpr float temp = 0.7978845608f;

    std::vector<float> result(size_input);

#pragma omp parallel for
    for (size_t i = 0; i < size_input; ++i) {
        float x = input[i];
        result[i] = 0.5f * x * (1.f + tanh(temp * (x + 0.044715f * x * x * x)));
    }

    return result;
}