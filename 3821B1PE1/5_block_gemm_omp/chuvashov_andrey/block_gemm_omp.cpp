#include "block_gemm_omp.h"

#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> result(n * n, 0.0f);
    
    auto countOfBlocks = n / SIZE;

    #pragma omp parallel for shared(a, b, c)
    for (int i = 0; i < countOfBlocks; i++) {
        for (int j = 0; j < countOfBlocks; j++) {
            for (int k = 0; k < countOfBlocks; k++) {
                for (int l = 0; l < SIZE; l++) {
                    for (int m = 0; m < SIZE; m++) {
                        float current = 0.0f;
                        for (int p = 0; p < SIZE; p++) {
                            current += a[(i * SIZE + l) * n + k * SIZE + p] *
                                b[(k * SIZE + p) * n + j * SIZE + m];
                        }
                        result[(i * SIZE + l) * n + j * SIZE + m] += current;
                    }
                }
            }
        }
    }
    return result;
}
