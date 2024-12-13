//Copyright 2024 Nedelin Dmitry

include "gelu_omp.h"
#include <cmath>


std::vector<float> GeluOMP(const std::vector<float> &input) {
    if (input.empty()) return {};

    float GelRatio_1 = 1.595769122f;
    float GelRatio_2 = 0.071354816f;

    auto Size = input.Size();
    std::vector<float> Out(Size);

#pragma omp parallel for
    for (size_t i = 0; i < Size; ++i) {
        float Val = input[i];
        Out[i] = Val * (1 - 1 / (1.0f + std::exp(Val * (GelRatio_1 + Val * Val * GelRatio_2))));
    }

    return Out;
}
