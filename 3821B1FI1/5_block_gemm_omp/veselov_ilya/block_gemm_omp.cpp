#include "block_gemm_omp.h"
#include <omp.h>
#include <algorithm>

void multiplyBlock(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, 
                   int n, int blockSize, int rowStart, int colStart, int innerStart) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < blockSize; ++i) {
        for (int j = 0; j < blockSize; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < blockSize; ++k) {
                sum += a[(rowStart + i) * n + (innerStart + k)] * b[(innerStart + k) * n + (colStart + j)];
            }
            c[(rowStart + i) * n + (colStart + j)] += sum;
        }
    }
}

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    int blockSize = 16;

    int blockCount = n / blockSize;

    std::vector<float> c(n * n, 0.0f);

    #pragma omp parallel for collapse(2)
    for (int blockRow = 0; blockRow < blockCount; ++blockRow) {
        for (int blockCol = 0; blockCol < blockCount; ++blockCol) {
            for (int innerBlock = 0; innerBlock < blockCount; ++innerBlock) {
                multiplyBlock(a, b, c, n, blockSize, blockRow * blockSize, blockCol * blockSize, innerBlock * blockSize);
            }
        }
    }

    return c;
}