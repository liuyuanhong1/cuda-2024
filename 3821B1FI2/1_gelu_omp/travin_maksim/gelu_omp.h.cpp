#include "gelu_omp.h"
#include <vector>
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    const float coeff = 0.044715f;

    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float inner = 1.59577f * (x + coeff * x * x * x);
        float tanh_val = std::tanh(inner);
        output[i] = 0.5f * x * (1.0f + tanh_val);
    }

    return output;
}