// Copyright (c) 2024 Moiseev Nikita
#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& matrix_a, const std::vector<float>& matrix_b, int dimension) {
    int total_elements = dimension * dimension;
    if (matrix_a.size() != total_elements || matrix_b.size() != total_elements) {
        return {};
    }

    std::vector<float> result_matrix(total_elements, 0.0f);

#pragma omp parallel for collapse(2)
    for (int row = 0; row < dimension; ++row) {
        for (int col = 0; col < dimension; ++col) {
            float element_sum = 0.0f;
            for (int k = 0; k < dimension; ++k) {
                element_sum += matrix_a[row * dimension + k] * matrix_b[k * dimension + col];
            }
            result_matrix[row * dimension + col] = element_sum;
        }
    }
    return result_matrix;
}