//Copyright Kutarin Aleksandr 2024

#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// CUDA kernel для вычисления GELU
__global__ void geluKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); // sqrt(2/pi) ≈ 0.79788
        output[idx] = x * cdf;
    }
}

// Основная функция GeluCUDA
std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int size = input.size();

    // Хост память
    float* h_input = nullptr;
    float* h_output = nullptr;

    // Выделение хост-памяти
    h_input = new float[size];
    h_output = new float[size];

    // Копирование данных во входной массив
    std::copy(input.begin(), input.end(), h_input);

    // Указатели на GPU
    float* d_input = nullptr;
    float* d_output = nullptr;

    // Выделение памяти на GPU
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Копирование данных с хоста на устройство
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Настройка размеров блоков и сетки
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Запуск ядра CUDA
    geluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);

    // Копирование результата обратно на хост
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Перенос данных в std::vector
    std::vector<float> result(h_output, h_output + size);

    // Очистка памяти
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    return result;
}
