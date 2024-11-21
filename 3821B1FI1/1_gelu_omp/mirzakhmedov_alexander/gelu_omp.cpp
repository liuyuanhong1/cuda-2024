#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> result(input.size());

#pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        result[i] = 0.5f * x * (1.0f + std::tanh((2.0f / M_PI) * (x + 0.044715f * std::pow(x, 3))));
    }

    return result;
}
