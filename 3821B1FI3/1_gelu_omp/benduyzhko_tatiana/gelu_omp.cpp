#include "gelu_omp.h"
#include <cmath>

const float sqrt2pi = 0.797884f;

std::vector<float> GeluOMP(const std::vector<float>& input) {
  std::vector<float> result(input);

#pragma omp parallel for
  for (int i = 0; i < result.size(); ++i) {
    const float x = result[i];
    result[i] = 0.5f * x *
                (1.0f + tanhf(sqrt2pi * x * (1.0f + 0.044715f * x * x)));
  }

  return result;
}
