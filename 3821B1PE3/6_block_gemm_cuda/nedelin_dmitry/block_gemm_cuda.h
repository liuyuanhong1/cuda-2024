// Copyright (c) 2024 Nedelin Dmitry

#ifndef __BLOCK_GEMM_CUDA_H
#define __BLOCK_GEMM_CUDA_H

#include <vector>

std::vector<float> BlockGemmCUDA(const float* matrixA,
                                 const float* matrixB,
                                 float* matrixC,int dimension);

#endif // __BLOCK_GEMM_CUDA_H