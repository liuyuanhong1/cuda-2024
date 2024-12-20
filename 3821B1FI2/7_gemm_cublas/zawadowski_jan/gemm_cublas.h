#ifndef __GEMM_CUBLAS_H
#define __GEMM_CUBLAS_H

#include <vector>
#include <cstdlib>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n);

#endif // __GEMM_CUBLAS_H