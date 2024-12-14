#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b, int size) {
    std::vector<float> c(size * size, 0.0f);
    int block_size = 16;  // cache line (64 bytes) / size of float (4 bytes)
    int num_blocks = size / block_size;

#pragma omp parallel for collapse(2)
    for (int i = 0; i < num_blocks; ++i) {
        for (int j = 0; j < num_blocks; ++j) {
            for (int k = 0; k < num_blocks; ++k) {
                for (int ii = 0; ii < block_size; ++ii) {
                    for (int jj = 0; jj < block_size; ++jj) {
                        float sum = 0.0f;
                        for (int kk = 0; kk < block_size; ++kk) {
                            sum += a[(i * block_size + ii) * size + (k * block_size + kk)] *
                                   b[(k * block_size + kk) * size + (j * block_size + jj)];
                        }
                        c[(i * block_size + ii) * size + (j * block_size + jj)] += sum;
                    }
                }
            }
        }
    }
    return c;
}
