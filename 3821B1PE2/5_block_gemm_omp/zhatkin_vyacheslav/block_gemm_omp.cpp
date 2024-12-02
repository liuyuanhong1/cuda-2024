#include "block_gemm_omp.h"
#include <omp.h>
#include <vector>
#include <cmath>
#include <stdexcept> 

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    // Result matrix initialized to zero
    std::vector<float> c(n * n, 0.0f);

    // Define block size
    int block_size = std::sqrt(n); // You can adjust block size for performance

    // Ensure block size divides n evenly
    if (n % block_size != 0) {
        throw std::invalid_argument("Matrix size n must be evenly divisible by block size.");
    }

    int num_blocks = n / block_size;

    // Parallelized block multiplication
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int bi = 0; bi < num_blocks; ++bi) {
        for (int bj = 0; bj < num_blocks; ++bj) {
            // Initialize block result
            for (int bk = 0; bk < num_blocks; ++bk) {
                // Perform block multiplication
                for (int i = bi * block_size; i < (bi + 1) * block_size; ++i) {
                    for (int j = bj * block_size; j < (bj + 1) * block_size; ++j) {
                        float sum = 0.0f;
                        for (int k = bk * block_size; k < (bk + 1) * block_size; ++k) {
                            sum += a[i * n + k] * b[k * n + j];
                        }
                        c[i * n + j] += sum;
                    }
                }
            }
        }
    }

    return c;
}