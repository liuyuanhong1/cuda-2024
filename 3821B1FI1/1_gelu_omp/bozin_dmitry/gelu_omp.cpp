// Copyright (c) 2024 Bozin Dmitry
#include "gelu_omp.h"

#include <cmath>
#include <chrono>
#include <iostream>
#include <ostream>
#include <random>
#include <cstdlib>


std::vector<float> GeluOMP(const std::vector<float> &input) {
  
  std::vector<float> result(input.size());
  constexpr float sqrt2_pi = 0.797885f;
  constexpr float coeff = 0.044715f;
  #pragma omp parallel for
  for(size_t i = 0;i < result.size();++i){
    float x = input[i]; 
    float tanh_arg = sqrt2_pi * (x + coeff * x * x * x);
    result[i] = 0.5f * x * (1.0f + tanh(tanh_arg));
  }
  return result;
}
