#include "gelu_omp.hpp"
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
        
        output_data[i] = x * (1.0f - 1.0f / (1.0f + std::exp(x * (GELU_COEF1 + x * x * GELU_COEF2)))); 
    }

    return output;
}