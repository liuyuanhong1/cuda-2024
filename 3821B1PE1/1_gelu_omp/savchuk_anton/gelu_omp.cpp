#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t size = input.size();
    std::vector<float> output(size);

    const float sqrt_2_over_pi = std::sqrt(2.0f / static_cast<float>(M_PI));
    const float coeff = 0.044715f;

#pragma omp parallel for
    for (long i = 0; i < static_cast<long>(size); ++i) {
        float x = input[i];
        float x_cubed = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
        float tanh_val = std::tanh(tanh_arg);
        output[i] = 0.5f * x * (1.0f + tanh_val);
    }

    return output;
}