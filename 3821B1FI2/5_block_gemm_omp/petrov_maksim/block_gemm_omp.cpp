#include <omp.h>
#include "block_gemm_omp.h"
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
   const std::vector<float>& b,
   int n) {
   const int blockSize = 16;

   std::vector<float> c(n * n, 0.0f);

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int blockRow = 0; blockRow < n; blockRow += blockSize) {  
        for (int blockCol = 0; blockCol < n; blockCol += blockSize) {
            for (int blockK = 0; blockK < n; blockK += blockSize) {
                
                for (int row = blockRow; row < blockRow + blockSize && row < n; ++row) {
                    for (int col = blockCol; col < blockCol + blockSize && col < n; ++col) {
                        float sum = 0.0f;
                        for (int k = blockK; k < blockK + blockSize && k < n; ++k) {
                            sum += a[row * n + k] * b[k * n + col];
                        }

                        c[row * n + col] += sum;
                    }
                }
            }
        }
    }
       return c;
   }