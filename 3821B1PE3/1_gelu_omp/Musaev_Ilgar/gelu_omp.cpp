// Copyright (c) 2024 Musaev Ilgar
#include "gelu_omp.h"

#include <omp.h>
#include <cmath>
#include <iostream>
#include <vector>

const float kTwoOverPi = sqrt(2.0f / M_PI); 

std::vector<float> GeluOMP(const std::vector<float>& input) {
  size_t size = input.size();
  std::vector<float> output(size); 

  #pragma omp parallel for
  for (std::size_t i = 0; i < size; i++) {
    float x = input[i];
    float temp = kTwoOverPi * (x + 0.044715f * x * x * x); 
    output[i] = 0.5f * x * (1.0f + tanh(temp));
  }

  return output;
}