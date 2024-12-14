/*When entering the following vector:
std::vector<float> input = {1.0, 2.0, 3.0, 4.0, 5.0};
The output values ​​were:
{0.841192, 1.9546, 2.99636, 3.99993, 5}
*/
#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

#ifndef MATH_PI
#define MATH_PI 3.14159265358979323846
#endif

std::vector<float> GeluOMP(const std::vector<float>& input) {
	size_t n = input.size();
	std::vector<float> output(n);

	const float sqrt_2_divide_pi = std::sqrt(2.0 / MATH_PI);

#pragma omp parallel for
	for (size_t i = 0; i < n; ++i) {
		float x = input[i];
		float xСubed = x * x * x;
		float tanhArg = sqrt_2_divide_pi * (x + 0.044715 * xСubed);
		float geluVal = 0.5f * x * (1.0f + std::tanh(tanhArg));
		output[i] = geluVal;
	}

	return output;
}