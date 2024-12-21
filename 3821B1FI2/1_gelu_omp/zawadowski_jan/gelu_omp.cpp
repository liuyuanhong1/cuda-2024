#include "gelu_omp.h"
#include <vector>
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float hyperTan = std::tanh(0.7978845608f * (x + 0.044715f * x * x * x));
        output[i] = 0.5f * x * (1.0f + hyperTan);
    }

    return output;
}