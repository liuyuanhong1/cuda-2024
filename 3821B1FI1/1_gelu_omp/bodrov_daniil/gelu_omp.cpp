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
    // Implement GELU function with omp
    size_t n = input.size();
    std::vector<float> result(n);
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        result[i] = 0.5 * input[i] * (1 + tanh(sqrt(2 / M_PI) * (input[i] + 0.044715 * pow(input[i], 3))));
    }
    return result;
}