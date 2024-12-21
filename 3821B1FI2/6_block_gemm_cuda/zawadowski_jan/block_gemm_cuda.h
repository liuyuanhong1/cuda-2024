#ifndef __BLOCK_GEMM_CUDA_H
#define __BLOCK_GEMM_CUDA_H

#include <cuda_runtime.h>
#include <vector>
#include <iostream>

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n);

#endif // __BLOCK_GEMM_CUDA_H