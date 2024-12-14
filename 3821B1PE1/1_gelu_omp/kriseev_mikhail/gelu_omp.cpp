#include "gelu_omp.h"
#include <cmath>

#define GELU_COEF1 1.595769122f
#define GELU_COEF2 0.071354816f

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    const int size = input.size();

    const float* input_data = input.data();
    float* output_data = output.data();

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        float x = input_data[i];
        float expon = std::exp(x * std::fma(GELU_COEF2, std::pow(x, 2), GELU_COEF1));
        output_data[i] = x * expon / (1.0f + expon);
        // output_data[i] = x * (1.0f - 1.0f / (1.0f + std::exp(x * (GELU_COEF1 + x * x * GELU_COEF2)))); 
    }

    return output;
}