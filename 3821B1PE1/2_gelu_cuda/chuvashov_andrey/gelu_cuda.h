// Copyright (c) 2024 Chuvashov Andrey
#ifndef __GELU_CUDA_H
#define __GELU_CUDA_H

#include <vector>

#define PER_C 0.044715f

std::vector<float> GeluCUDA(const std::vector<float>& input);

#endif // __GELU_CUDA_H
