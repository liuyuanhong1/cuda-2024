#include <cstdlib>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "block_gemm_cuda.h"

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int size) {
    
}