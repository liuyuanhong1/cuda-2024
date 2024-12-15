#include "gelu_omp.h"

#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
	std::vector<float> result(input);
	constexpr float twooverpi = 0.7978845608028653;
#pragma omp parallel for
	for (int i = 0; i < result.size(); i++) {
		float x = result[i];
		result[i] = 0.5f * x * (1.f + tanhf(twooverpi * x * (1.0f + 0.044715f * x * x)));
	}
    return result;
}
