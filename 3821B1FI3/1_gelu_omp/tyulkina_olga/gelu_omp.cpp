
#include "gelu_omp.h"

#include <cmath>

std::vector<float> GeluOMP(const std::vector<float> &input) {
  if (input.empty()) return {};

  std::vector<float> output(input.size());

  const float constOne = 0.044715;
  const float constTwo = std::sqrt(2.0 / M_PI);

#pragma omp parallel for
  for (size_t i = 0; i < input.size(); i++) {
    float x = input[i];
    float tanh_arg = constTwo * (x + constOne * (x * x * x));
    output[i] = 0.5 * x * (1.0 + std::tanh(tanh_arg));
  }

  return output;
}