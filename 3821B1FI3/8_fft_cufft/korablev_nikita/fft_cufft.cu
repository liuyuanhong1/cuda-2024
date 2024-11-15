// Copyright (c) 2024 Korbalev Nikita
#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    std::vector<float> output(input.size(), 0.0f);

    cufftComplex* d_input;
    cufftComplex* d_output;
    cudaMalloc(&d_input, batch * n * sizeof(cufftComplex));
    cudaMalloc(&d_output, batch * n * sizeof(cufftComplex));

    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD);
    cufftExecC2C(plan, d_output, d_input, CUFFT_INVERSE);

    float normalizationFactor = 1.0f / n;
    cufftComplex* d_normalized;
    cudaMalloc(&d_normalized, batch * n * sizeof(cufftComplex));
    cufftExecC2C(plan, d_input, d_normalized, CUFFT_FORWARD);

    cudaMemcpy(output.data(), d_normalized, input.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_normalized);

    cufftDestroy(plan);
    return output;
}
