// Copyright (c) 2024 Moiseev Nikita
#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float>& matrix_a, const std::vector<float>& matrix_b, int matrix_size) {
    auto total_elements = matrix_size * matrix_size;
    if (matrix_a.size() != total_elements || matrix_b.size() != total_elements) return {};

    std::vector<float> matrix_c(total_elements, 0.0f);
    const auto block_size = 8;
    auto num_blocks = matrix_size / block_size;

#pragma omp parallel for shared(matrix_a, matrix_b, matrix_c)
    for (int block_row = 0; block_row < num_blocks; ++block_row)
        for (int block_col = 0; block_col < num_blocks; ++block_col)
            for (int block_k = 0; block_k < num_blocks; ++block_k)
                for (int row = block_row * block_size; row < (block_row + 1) * block_size; ++row)
                    for (int col = block_col * block_size; col < (block_col + 1) * block_size; ++col)
                        for (int k = block_k * block_size; k < (block_k + 1) * block_size; ++k)
                            matrix_c[row * matrix_size + col] += matrix_a[row * matrix_size + k] * matrix_b[k * matrix_size + col];

    return matrix_c;
}
