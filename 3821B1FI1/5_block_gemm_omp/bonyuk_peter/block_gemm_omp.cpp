#include "block_gemm_omp.h"
#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float> &a,
	const std::vector<float> &b, int size) {
	std::vector<float> c(size * size, 0.0f);
	int size_block = 16;
	int n_block = size / size_block;

#pragma omp parallel for collapse(2)
	for (int i = 0; i < n_block; ++i) {
		for (int j = 0; j < n_block; ++j) {
			for (int k = 0; k < n_block; ++k) {
				for (int i1 = 0; i1 < size_block; ++i1) {
					for (int j1 = 0; j1 < size_block; ++j1) {
						float s = 0.0f;
						for (int k1 = 0; k1 < size_block; ++k1) {
							s += a[(i * size_block + i1) * size + (k * size_block + k1)] *
								b[(k * size_block + k1) * size + (j * size_block + j1)];
						}
						c[(i * size_block + i1) * size + (j * size_block + j1)] += s;
					}
				}
			}
		}
	}
	return c;
}