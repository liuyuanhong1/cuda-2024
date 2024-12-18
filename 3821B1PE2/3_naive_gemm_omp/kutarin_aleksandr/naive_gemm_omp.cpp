//Copyright Kutarin Aleksandr 2024

#include "naive_gemm_omp.h"
#include <omp.h> // Include OpenMP header for parallelization

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    // Resultant matrix initialized to zero
    std::vector<float> c(n * n, 0.0f);

    // Perform matrix multiplication with OpenMP
#pragma omp parallel for collapse(2) // Parallelize nested loops
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }

    return c;
}
