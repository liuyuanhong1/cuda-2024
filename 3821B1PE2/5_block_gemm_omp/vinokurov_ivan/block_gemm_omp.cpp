//  Copyright (c) 2024 Vinokurov Ivan
#include "block_gemm_omp.h"

#define BLOCK_SIZE 16

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    std::vector<float> output(n * n, 0.0f);
    int totalBlocks = n / BLOCK_SIZE;

#pragma omp parallel for collapse(2)
    for (int blockRow = 0; blockRow < totalBlocks; ++blockRow) {
        for (int blockCol = 0; blockCol < totalBlocks; ++blockCol) {
            for (int blockIndex = 0; blockIndex < totalBlocks; ++blockIndex) {
                for (int localRow = 0; localRow < BLOCK_SIZE; ++localRow) {
                    for (int localCol = 0; localCol < BLOCK_SIZE; ++localCol) {
                        float localSum = 0.0f;
                        for (int localIndex = 0; localIndex < BLOCK_SIZE; ++localIndex) {
                            localSum += a[(blockRow * BLOCK_SIZE + localRow) * n + blockIndex * BLOCK_SIZE + localIndex] *
                                b[(blockIndex * BLOCK_SIZE + localIndex) * n + blockCol * BLOCK_SIZE + localCol];
                        }
                        output[(blockRow * BLOCK_SIZE + localRow) * n + blockCol * BLOCK_SIZE + localCol] += localSum;
                    }
                }
            }
        }
    }
    return output;
}