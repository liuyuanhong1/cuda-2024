#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> PerformNaiveGEMM(const std::vector<float>& matrixA,
                                     const std::vector<float>& matrixB, int dimension) {
  
    const int totalElements = dimension * dimension;

    
    if (matrixA.size() != totalElements || matrixB.size() != totalElements) {
        return {};  
    }

    
    std::vector<float> resultMatrix(totalElements, 0.0f);

    
    #pragma omp parallel for
    for (int row = 0; row < dimension; ++row) {
        for (int col = 0; col < dimension; ++col) {
            for (int idx = 0; idx < dimension; ++idx) {
                resultMatrix[row * dimension + col] += matrixA[row * dimension + idx] * matrixB[idx * dimension + col];
            }
        }
    }

    return resultMatrix;
}
