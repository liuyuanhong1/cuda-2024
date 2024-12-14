#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t m = input.size();
    std::vector<float> out(m);

    const float sqrt = std::sqrt(2.0 / M_PI);

#pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        float x = input[i];
        float x_c = x * x * x;
        float arg_tan = sqrt * (x + 0.044715 * x_c);
        float value = 0.5f * x * (1.0f + std::tanh(arg_tan));
        out[i] = value;
    }

    return out;
}