#include "block_gemm_omp.h"
#include <omp.h>
#include <algorithm>

void multiplyBlock(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, 
                   int n, int blockSize, int rowStart, int colStart, int innerStart) {
    std::vector<float> blockA(blockSize * blockSize);
    std::vector<float> blockB(blockSize * blockSize);
    std::vector<float> blockC(blockSize * blockSize, 0.0f);

    for (int i = 0; i < blockSize; ++i) {
        for (int j = 0; j < blockSize; ++j) {
            blockA[i * blockSize + j] = a[(rowStart + i) * n + (innerStart + j)];
            blockB[i * blockSize + j] = b[(innerStart + j) * n + (colStart + i)];
        }
    }

    for (int i = 0; i < blockSize; ++i) {
        #pragma omp simd
        for (int j = 0; j < blockSize; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < blockSize; ++k) {
                sum += blockA[i * blockSize + k] * blockB[k * blockSize + j];
            }
            blockC[i * blockSize + j] += sum;
        }
    }

    for (int i = 0; i < blockSize; ++i) {
        for (int j = 0; j < blockSize; ++j) {
            c[(rowStart + i) * n + (colStart + j)] += blockC[i * blockSize + j];
        }
    }
}

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    int blockSize = 16;
    if (n > 64) {
        blockSize = 32;
    }
    if (n > 128) {
        blockSize = 64;
    }

    int blockCount = n / blockSize;

    std::vector<float> c(n * n, 0.0f);

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int blockRow = 0; blockRow < blockCount; ++blockRow) {
        for (int blockCol = 0; blockCol < blockCount; ++blockCol) {
            for (int innerBlock = 0; innerBlock < blockCount; ++innerBlock) {
                multiplyBlock(a, b, c, n, blockSize, blockRow * blockSize, blockCol * blockSize, innerBlock * blockSize);
            }
        }
    }

    return c;
}
