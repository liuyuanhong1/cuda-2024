#include "gelu_omp.h"

#include <cmath>


std::vector<float> GeluOMP(const std::vector<float>& input) {
   
    if (input.empty()) return {};

  
    constexpr float constOne = 1.595769122f; // 2 * sqrt(2 / PI)
    constexpr float constTwo = constOne * 0.044715f;

   
    std::vector<float> output(input.size());

    
    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float expArg = x * (constOne + x * x * constTwo); 
        output[i] = x - x / (1.0f + std::exp(expArg)); 
    }

    return output;
}
