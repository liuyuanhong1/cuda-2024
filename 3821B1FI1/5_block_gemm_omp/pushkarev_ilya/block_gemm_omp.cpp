#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b, int n) {

    int block_size = 16;
    int blocks_num = n / block_size;
    
    std::vector<float> c(n * n, 0.0f);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < blocks_num; ++i) 
    {
        for (int j = 0; j < blocks_num; ++j) 
        {
            for (int r = 0; r < blocks_num; ++r) 
            {
                for (int ii = 0; ii < block_size; ++ii) 
                {
                    for (int jj = 0; jj < block_size; ++jj) 
                    {
                        float sum = 0.0f;
                        for (int rr = 0; rr < block_size; ++rr) 
                        {
                            sum += a[(i * block_size + ii) * n + (r * block_size + rr)] *
                                   b[(r * block_size + rr) * n + (j * block_size + jj)];
                        }
                        c[(i * block_size + ii) * n + (j * block_size + jj)] += sum;
                    }
                }
            }
        }
    }
    return c;
}