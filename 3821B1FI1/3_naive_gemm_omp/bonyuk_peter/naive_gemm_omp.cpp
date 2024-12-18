/*When entering the following vector:
int size = 3;
std::vector<float> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
std::vector<float> b = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
The output values ​​were:
30 24 18
96 69 54
177 114 90
*/

#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
	const std::vector<float>& b, int size) {

	std::vector<float> c(size * size, 0.0f);
	float* c1;
	const float* b1;
	float sl;

#pragma omp parallel for private(sl, c1, b1)
	for (int m = 0; m < size; m++) {
		c1 = &c[m * size];
		for (int k = 0; k < size; k++) {
			sl = a[m * size + k];
			b1 = &b[size * k];
			for (int n = 0; n < size; n += 2) {
				c1[n] += sl * b1[n];
				c1[n + 1] += sl * b1[n + 1];
			}
		}
	}
	return c;
}