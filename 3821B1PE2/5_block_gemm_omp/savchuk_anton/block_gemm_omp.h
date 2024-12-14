#ifndef __BLOCK_GEMM_OMP_H
#define __BLOCK_GEMM_OMP_H

#include <vector>

// Прототип функции для блочного перемножения двух квадратных матриц
std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n);

#endif  // __BLOCK_GEMM_OMP_H
