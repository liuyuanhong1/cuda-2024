#include "gelu_omp.h"
#include <cmath>      // Для std::tanh и std::sqrt
#include <omp.h>      // Для OpenMP

#define M_PI 3.14

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size()); // Результирующий вектор такого же размера, как входной

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(input.size()); ++i) { // Используйте int вместо size_t
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }

    return output;
}