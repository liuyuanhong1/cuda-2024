#include "gelu_omp.h"
#include <omp.h>
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    if (input.empty()) return {};
    std::vector<float> result(input.size());
    const size_t n = input.size();

#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        // 2 * sqrt(2.0f / 3.1415926535f) = 1.59577f
        float tanh_arg = 1.59577f * (x + 0.044715f * x * x * x);
        float exp1 = std::exp(tanh_arg);

        // tanh(x) = (e^2x - 1)/(e^2x + 1)

        result[i] = 0.5f * x * (1.0f + (exp1 - 1)/(exp1 + 1));
    }
    return result;
}