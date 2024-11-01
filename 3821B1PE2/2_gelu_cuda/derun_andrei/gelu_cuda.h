#ifndef __GELU_CUDA_H
#define __GELU_CUDA_H

#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

std::vector<float> GeluCUDA(const std::vector<float> &input);

#endif // __GELU_CUDA_H