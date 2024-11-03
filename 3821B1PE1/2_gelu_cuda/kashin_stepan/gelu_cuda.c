// Copyright (c) 2024 Kashin Stepan

#ifndef GELU_CUDA_H
#define GELU_CUDA_H

#include <vector>

std::vector<float> ComputeGeluCUDA(const std::vector<float>& input);

#endif  // GELU_CUDA_H