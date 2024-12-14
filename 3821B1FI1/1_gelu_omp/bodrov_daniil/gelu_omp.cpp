/* The Gaussian Error Linear Unit (GELU) is an activation function frequently used in Deep Neural Networks (DNNs) and can be thought of as a smoother ReLU.
To approximate GELU function, use the following formula:
GELU(x) =  $0.5x(1 + tanh(\sqrt{2 / \pi}(x + 0.044715 * x^3)))$

Implement the function with the following interface in C++:
```cpp
std::vector<float> GeluOMP(const std::vector<float>& input);
```
Size of result vector should be the same as for `input`. Use OpenMP technology to make your function parallel & fast.
*/
// Includes libs for math and omp
#include <cmath>
#include <omp.h>
#include "gelu_omp.h"

// Define M_PI
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    const float sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
    const float coeff = 0.044715f;

    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float x_cubed = x * x * x;
        float tanh_value = std::tanh(sqrt_2_over_pi * (x + coeff * x_cubed));
        output[i] = 0.5f * x * (1.0f + tanh_value);
    }

    return output;
}