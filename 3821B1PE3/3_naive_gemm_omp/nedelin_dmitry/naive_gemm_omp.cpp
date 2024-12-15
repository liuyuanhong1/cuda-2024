// Copyright (c) 2024 Nedelin Dmitry

#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& matrixA,
    const std::vector<float>& matrixB,
    int dimension) {
    if (matrixA.size() != dimension * dimension || matrixB.size() != dimension * dimension) {
        return {};
    }

    std::vector<float> matrixC(dimension * dimension, 0.0f);

#pragma omp parallel for collapse(2)
    for (int row = 0; row < dimension; ++row) {
        for (int col = 0; col < dimension; ++col) {
            float elementSum = 0.0f;
            for (int inner = 0; inner < dimension; ++inner) {
                elementSum += matrixA[row * dimension + inner] * matrixB[inner * dimension + col];
            }
            matrixC[row * dimension + col] = elementSum;
        }
    }

    return matrixC;
}

