// Copyright (c) 2024 Kashin Stepan
// Разрешаю копировать и взаимствовать, если указан оригинальный автор: "// Copyright (c) 2024 Kashin Stepan -> Ivanov Ivan"
#include "gelu_omp.h"

#include <math.h>
#include <vector>
#include <omp.h>

// sqrt(pi/2)
const float a = 0.7978845608f;
// cdf_coefficient
const float b = 0.044715f;

std::vector<float> GeluOMP(const std::vector<float>& input) {
  const size_t range = input.size();
  std::vector<float> answer(range);

  #pragma omp parallel for
  for (size_t i = 0; i < range; i++) {
    float x = input[i];
    answer[i] = 0.5f * x * (1.0f + tanhf(a * (x + b * x * x * x)));
  }

  return answer;
}