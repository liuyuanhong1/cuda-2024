#include "gelu_omp.h"
#include <cmath> 
#include <omp.h> 

#ifndef M_PI
#define M_PI 3.14159265358979323846 
#endif

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(input.size()); ++i) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }

    return output;
}
