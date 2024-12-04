#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
    }

// Макрос для проверки ошибок cuFFT
#define CUFFT_CHECK(err) \
    if (err != CUFFT_SUCCESS) { \
        throw std::runtime_error(std::string("cuFFT Error: ") + cufftGetErrorString(err)); \
    }

// Функция для получения строкового описания ошибок cuFFT
const char* cufftGetErrorString(cufftResult error) {
    switch(error) {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";
        default:
            return "Unknown cuFFT error";
    }
}

// CUDA-ядро для нормализации данных на устройстве
__global__ void normalize(cufftComplex* data, int size, float norm_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx].x *= norm_factor;
        data[idx].y *= norm_factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    // Проверка корректности размера входных данных
    if (input.size() % (2 * batch) != 0) {
        throw std::invalid_argument("Размер входного массива не соответствует формату (real, imaginary) для заданного batch.");
    }

    // Вычисление размера одного сигнала
    int n = input.size() / (2 * batch);

    // Общее количество комплексных чисел
    int total_elements = n * batch;

    // Размер данных в байтах
    size_t bytes = sizeof(cufftComplex) * total_elements;

    // Указатель на данные на устройстве
    cufftComplex* d_data = nullptr;

    // Выделение памяти на устройстве
    CUDA_CHECK(cudaMalloc((void**)&d_data, bytes));

    // Копирование данных с хоста на устройство
    CUDA_CHECK(cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice));

    // Создание дескриптора cuFFT
    cufftHandle plan;

    // Создание плана для FFT
    CUFFT_CHECK(cufftPlan1d(&plan, n, CUFFT_C2C, batch));

    // Выполнение прямого FFT (in-place)
    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

    // Выполнение обратного FFT (in-place)
    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));

    // Нормализация результата на устройстве
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    float norm_factor = 1.0f / static_cast<float>(n);

    normalize<<<blocksPerGrid, threadsPerBlock>>>(d_data, total_elements, norm_factor);

    // Проверка на ошибки после запуска ядра
    CUDA_CHECK(cudaGetLastError());

    // Синхронизация устройства
    CUDA_CHECK(cudaDeviceSynchronize());

    // Выделение памяти для результата на хосте
    std::vector<float> output(2 * total_elements);

    // Копирование результата с устройства на хост
    CUDA_CHECK(cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost));

    // Освобождение плана
    CUFFT_CHECK(cufftDestroy(plan));

    // Освобождение памяти на устройстве
    CUDA_CHECK(cudaFree(d_data));

    return output;
}