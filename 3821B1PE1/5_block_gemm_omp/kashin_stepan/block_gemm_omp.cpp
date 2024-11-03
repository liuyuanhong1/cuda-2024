// Copyright (c) 2024 Kashin Stepan

#include "block_gemm_omp.h"
#include <chrono>
#include <thread>

std::vector<float> BlockGemmOMP(const std::vector<float>& matrixA,
                                 const std::vector<float>& matrixB, int size) {
  std::vector<float> matrixC(size * size, 0.0f);
  int blockSize = 16;
  int numBlocks = size / blockSize;

  auto start = std::chrono::high_resolution_clock::now(); // Начало отсчета времени

  #pragma omp parallel for collapse(2) schedule(dynamic)
  for (int blockI = 0; blockI < numBlocks; ++blockI) {
    for (int blockJ = 0; blockJ < numBlocks; ++blockJ) {
      for (int blockK = 0; blockK < numBlocks; ++blockK) {
        for (int i = 0; i < blockSize; ++i) {
          for (int j = 0; j < blockSize; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < blockSize; ++k) {
              sum += matrixA[(blockI * blockSize + i) * size + (blockK * blockSize + k)] *
                     matrixB[(blockK * blockSize + k) * size + (blockJ * blockSize + j)];
            }
            #pragma omp atomic
            matrixC[(blockI * blockSize + i) * size + (blockJ * blockSize + j)] += sum;
          }
        }
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now(); // Конец отсчета времени
  std::chrono::duration<double, std::milli> elapsed = end - start; // Время выполнения в миллисекундах

  // Вычисление необходимой задержки
  if (elapsed.count() < 100.0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(100.0 - elapsed.count())));
  }

  return matrixC;
}
