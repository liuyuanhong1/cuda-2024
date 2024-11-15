// Copyright (c) 2024 Chuvashov Andrey
#include "gelu_omp.h"

#include <math.h>
#include <omp.h>

const float pi_c = sqrt(2.0f / (2 * asin(1.0f))); // sqrt(2 / pi)
const float par_c= 0.044715f; // coefficient in parentheses

std::vector<float> GeluOMP(const std::vector<float>& input) {

    const size_t length = input.size();
    std::vector<float> result(length);

    #pragma omp parallel for
    for (size_t i = 0; i < length; i++) {
        float x = input[i];
        result[i] = 0.5f * x * (1.0f + tanhf(pi_c * (x + par_c * x * x * x)));
    }
    return result;
}
