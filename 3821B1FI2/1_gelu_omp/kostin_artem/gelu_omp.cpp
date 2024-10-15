#include <cmath>
#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
    int s = input.size();
    std::vector<float> output (s);
    if (s == 0) return output;
    constexpr float y = 0.797885; // sqrt(2.0 / M_PI)
    constexpr float w = 0.0356774; // y * 0.044715
#pragma omp parallel for
    for (int i = 0; i < s; i++) {
        output[i] = input[i] * ((1.0 / (1.0 + exp(-2.0 * (input[i] * (y + w * input[i] * input[i])))))); // tanh = (2.0 / (1.0 + exp(-2.0 * x))) - 1.0
    }
    return output;
}