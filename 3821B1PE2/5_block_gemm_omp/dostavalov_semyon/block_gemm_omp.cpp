// Copyright (c) 2024 Dostavalov Semyon

#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float>& matA,
    const std::vector<float>& matB, int dim) {
    std::vector<float> matC(dim * dim, 0.0f);
    int blockDim = 16;
    int numBlocks = dim / blockDim;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int blockRow = 0; blockRow < numBlocks; ++blockRow) {
        for (int blockCol = 0; blockCol < numBlocks; ++blockCol) {
            for (int blockDepth = 0; blockDepth < numBlocks; ++blockDepth) {
                for (int row = 0; row < blockDim; ++row) {
                    for (int col = 0; col < blockDim; ++col) {
                        float accumulation = 0.0f;
                        for (int depth = 0; depth < blockDim; ++depth) {
                            accumulation += matA[(blockRow * blockDim + row) * dim + (blockDepth * blockDim + depth)] *
                                matB[(blockDepth * blockDim + depth) * dim + (blockCol * blockDim + col)];
                        }
#pragma omp atomic
                        matC[(blockRow * blockDim + row) * dim + (blockCol * blockDim + col)] += accumulation;
                    }
                }
            }
        }
    }

    return matC;
}
