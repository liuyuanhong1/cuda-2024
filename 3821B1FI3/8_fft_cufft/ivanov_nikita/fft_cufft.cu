// Copyright (c) 2024 Ivanov Nikita
#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    std::vector<float> output(input.size(), 0.0f);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftComplex* d_input;
    cufftComplex* d_output;
    cudaMalloc((void**)&d_input, batch * n * sizeof(cufftComplex));
    cudaMalloc((void**)&d_output, batch * n * sizeof(cufftComplex));

    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

    cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD);

    cufftExecC2C(plan, d_output, d_input, CUFFT_INVERSE);

    float normalization_factor = 1.0f / n;
    cufftComplex* d_normalized_output;
    cudaMalloc((void**)&d_normalized_output, batch * n * sizeof(cufftComplex));
    cufftExecC2C(plan, d_input, d_normalized_output, CUFFT_FORWARD);

    cudaMemcpy(output.data(), d_normalized_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_normalized_output);

    cufftDestroy(plan);

    return output;
}
