// Copyright (c) 2024 Kulagin Aleksandr
#include "gelu_omp.h"

#define _USE_MATH_DEFINES
#include <omp.h>
#include <cmath>

static const float precalc_c_1 = std::sqrt(2.0f / M_PIf);

std::vector<float> GeluOMP(const std::vector<float>& input) {
  if (input.empty()) {
    return std::vector<float>();
  }
  const std::vector<float>::size_type sz = input.size();
  std::vector<float> res(sz);
#pragma omp parallel for schedule(static)
  for (std::vector<float>::size_type i = 0; i < sz; i++) {
    const float& x = input[i];
    res[i] = 0.5f * x * (1.0f + std::tanh(precalc_c_1 * ( x + 0.044715f * (x * x * x) )));
  }
  return res;
}
