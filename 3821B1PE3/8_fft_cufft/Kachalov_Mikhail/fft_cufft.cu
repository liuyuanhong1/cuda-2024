// Copyright 2024 Kachalov Mikhail
#include "fft_cufft.h"
#include <cufft.h>
#include <iostream>

std::vector<float> FffCUFFT(const std::vector<float> &input, int batch)
{
    int n = input.size() / (2 * batch);
    std::vector<float> output(input.size(), 0.0f);

    cufftHandle plan;
    cufftComplex *d_input, *d_output;
    cudaMalloc((void **)&d_input, input.size() * sizeof(float));
    cudaMalloc((void **)&d_output, input.size() * sizeof(float));
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cufftPlanMany(&plan, 1, &n, NULL, 1, n, NULL, 1, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD);
    cufftExecC2C(plan, d_output, d_output, CUFFT_INVERSE);
    cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < output.size(); i++)
    {
        output[i] /= n;
    }

    cufftDestroy(plan);
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}