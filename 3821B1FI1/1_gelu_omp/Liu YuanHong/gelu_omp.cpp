#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);

    const float sqrt_2_over_pi = std::sqrt(2.0 / M_PI);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        float x_cubed = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed);
        float gelu_val = 0.5f * x * (1.0f + std::tanh(tanh_arg));
        output[i] = gelu_val;
    }

    return output;
}
