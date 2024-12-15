// Copyright (c) 2024 Nedelin Dmitry

#include "block_gemm_omp.h"

#define BLOCK_SIZE 16

std::vector<float> BlockGemmOMP(const std::vector<float>& matrixA,
    const std::vector<float>& matrixB,
    int dimension) {
    std::vector<float> matrixC(dimension * dimension, 0.0f);
    int totalBlocks = dimension / BLOCK_SIZE;

#pragma omp parallel for collapse(2)
    for (int blockRow = 0; blockRow < totalBlocks; ++blockRow) {
        for (int blockCol = 0; blockCol < totalBlocks; ++blockCol) {
            for (int blockIndex = 0; blockIndex < totalBlocks; ++blockIndex) {
                for (int localRow = 0; localRow < BLOCK_SIZE; ++localRow) {
                    for (int localCol = 0; localCol < BLOCK_SIZE; ++localCol) {
                        float localSum = 0.0f;
                        for (int localIndex = 0; localIndex < BLOCK_SIZE; ++localIndex) {
                            localSum += matrixA[(blockRow * BLOCK_SIZE + localRow) * dimension + blockIndex * BLOCK_SIZE + localIndex] *
                                matrixB[(blockIndex * BLOCK_SIZE + localIndex) * dimension + blockCol * BLOCK_SIZE + localCol];
                        }
                        matrixC[(blockRow * BLOCK_SIZE + localRow) * dimension + blockCol * BLOCK_SIZE + localCol] += localSum;
                    }
                }
            }
        }
    }
    return matrixC;
}
