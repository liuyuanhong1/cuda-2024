// Copyright (c) 2024 Prokofev Kirill
#include "gelu_omp.h"

#include <cmath>
#include <chrono>
#include <iostream>
#include <ostream>
#include <random>
#include <cstdlib>


/*
std::vector<float> GenVec(const int size){
    std::vector<float> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for(size_t i = 0;i < vec.size();++i){
        vec[i] = dist(gen);
    }
    return vec;
}
*/

float fast_tanh(float x) {
    if (x < -3.0f) return -1.0f;
    if (x > 3.0f) return 1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}



std::vector<float> GeluOMP(const std::vector<float> &input) {
  
  std::vector<float> result(input.size());
  constexpr float sqrt2_pi = 0.797885f;
  constexpr float coeff = 0.044715f;
  #pragma omp parallel for
  for(size_t i = 0;i < result.size();++i){
    float x = input[i]; 
    float tanh_arg = sqrt2_pi * (x + coeff * x * x * x);
    result[i] = 0.5f * x * (1.0f + fast_tanh(tanh_arg));
  }
  return result;
}


/*
int main() {
    
    //std::vector<float> a {0.1,2.4,5.4,3.2};

    std::vector<float> a = GenVec(134217728);
    
    // Performance Measuring
    auto start = std::chrono::high_resolution_clock::now();
    auto c = GeluOMP(a);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration = end - start;

    std::cout << "Time taken: " << duration.count() << " s" << std::endl;

    return 0;

}
*/
