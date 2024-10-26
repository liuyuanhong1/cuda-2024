#ifndef __GEMM_CUBLAS_H
#define __GEMM_CUBLAS_H

#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cassert>

std::vector<float> GemmCUBLAS(const std::vector<float> &a,
                              const std::vector<float> &b,
                              int n);

#endif // __GEMM_CUBLAS_H
