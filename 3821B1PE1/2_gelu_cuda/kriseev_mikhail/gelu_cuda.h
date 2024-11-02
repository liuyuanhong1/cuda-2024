#ifndef __GELU_CUDA_H
#define __GELU_CUDA_H

#include <vector>

#define GELU_COEF1 1.595769122f
#define GELU_COEF2 0.071354816f

std::vector<float> GeluCUDA(const std::vector<float>& input);

#endif // __GELU_CUDA_H