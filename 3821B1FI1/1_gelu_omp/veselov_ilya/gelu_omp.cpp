#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        float gelu_val = 0.5 * x * (1 + std::tanh(std::sqrt(2 / M_PI) * (x + 0.044715 * std::pow(x, 3))));
        output[i] = gelu_val;
    }

    return output;
}
