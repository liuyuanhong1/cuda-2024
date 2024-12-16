// Copyright (c) 2024 Khodyrev Fedor
#include "gelu_omp.h"

#include <omp.h>
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    if (input.empty()) {
		return {};
	}
    const float const_gelu = 0.044715f;
    auto size_of_input = input.size();
    std::vector<float> output(size_of_input);

#pragma omp parallel for
  for (size_t i = 0; i < size_of_input; ++i) {
    float tmp = input[i];
    output[i] = 0.5f * tmp * (1.0f + tanh(sqrt(2.0f / M_PI)
     * (tmp + const_gelu * tmp * tmp * tmp)));
  }

  return output;
}