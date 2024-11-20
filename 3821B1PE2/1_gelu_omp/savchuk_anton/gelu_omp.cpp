#include "gelu_omp.h"
#include <cmath> // Для функций tanh и sqrt
#include <omp.h> // Для OpenMP

#ifndef M_PI
#define M_PI 3.14159265358979323846 // Определение числа π
#endif

std::vector<float> GeluOMP(const std::vector<float>& input) {
    // Инициализация вектора результата такого же размера, как и input
    std::vector<float> output(input.size());

    // Параллельный цикл для вычисления GELU
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(input.size()); ++i) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }

    return output;
}
