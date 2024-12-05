#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);

    if (input.size() != 2 * n * batch) {
        throw std::invalid_argument("Input size must be 2 * n * batch, where n is the number of elements per signal.");
    }

    cufftComplex* d_input;
    cufftComplex* d_output;
    size_t input_size = 2 * n * batch * sizeof(float);

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, input_size);

    cudaMemcpy(d_input, input.data(), input_size, cudaMemcpyHostToDevice);

    cufftHandle plan_forward;
    if (cufftPlan1d(&plan_forward, n, CUFFT_C2C, batch) != CUFFT_SUCCESS) {
        throw std::runtime_error("CUFFT plan creation for forward transform failed.");
    }

    if (cufftExecC2C(plan_forward, d_input, d_output, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        throw std::runtime_error("CUFFT execution for forward FFT failed.");
    }

    cufftHandle plan_inverse;
    if (cufftPlan1d(&plan_inverse, n, CUFFT_C2C, batch) != CUFFT_SUCCESS) {
        throw std::runtime_error("CUFFT plan creation for inverse transform failed.");
    }

    if (cufftExecC2C(plan_inverse, d_output, d_output, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        throw std::runtime_error("CUFFT execution for inverse FFT failed.");
    }

    int num_elements = 2 * n * batch;
    float norm_factor = 1.0f / static_cast<float>(n);
    cudaDeviceSynchronize();

    for (int i = 0; i < num_elements; i++) {
        if (i % 2 == 0) {
            d_output[i].x *= norm_factor;
        } else {
            d_output[i].y *= norm_factor;
        }
    }

    std::vector<float> result(input.size());
    cudaMemcpy(result.data(), d_output, input_size, cudaMemcpyDeviceToHost);

    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);
    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}
