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

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    // Проверка корректности размера входных данных
    if (input.size() % (2 * batch) != 0) {
        throw std::invalid_argument("Размер входного массива не соответствует формату (real, imaginary) для заданного batch.");
    }

    // Вычисление размера одного сигнала
    int n = input.size() / (2 * batch);

    // Размер данных в байтах
    size_t bytes = sizeof(float) * 2 * n * batch;

    // Указатели на устройства
    cufftComplex *d_input = nullptr;
    cufftComplex *d_forward = nullptr;
    cufftComplex *d_inverse = nullptr;

    // Выделение памяти на устройстве
    CUDA_CHECK(cudaMalloc((void**)&d_input, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_forward, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_inverse, bytes));

    // Копирование данных с хоста на устройство
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

    // Создание дескриптора cuFFT
    cufftHandle plan_forward;
    cufftHandle plan_inverse;

    // План для прямого FFT
    CUFFT_CHECK(cufftPlan1d(&plan_forward, n, CUFFT_C2C, batch));

    // План для обратного FFT
    CUFFT_CHECK(cufftPlan1d(&plan_inverse, n, CUFFT_C2C, batch));

    // Выполнение прямого FFT: d_input -> d_forward
    CUFFT_CHECK(cufftExecC2C(plan_forward, d_input, d_forward, CUFFT_FORWARD));

    // Выполнение обратного FFT: d_forward -> d_inverse
    CUFFT_CHECK(cufftExecC2C(plan_inverse, d_forward, d_inverse, CUFFT_INVERSE));

    // Освобождение планов
    CUFFT_CHECK(cufftDestroy(plan_forward));
    CUFFT_CHECK(cufftDestroy(plan_inverse));

    // Выделение памяти для результата на хосте
    std::vector<float> output(2 * n * batch);

    // Копирование результата с устройства на хост
    CUDA_CHECK(cudaMemcpy(output.data(), d_inverse, bytes, cudaMemcpyDeviceToHost));

    // Освобождение памяти на устройстве
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_forward));
    CUDA_CHECK(cudaFree(d_inverse));

    // Нормализация результата
    for(int i = 0; i < 2 * n * batch; ++i) {
        output[i] /= static_cast<float>(n);
    }

    return output;
}