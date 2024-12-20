#include <cufft.h>
#include <iostream>

#include "fft_cufft.h"

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) 
{
    int n = input.size() / (2 * batch);

    cufftComplex* device_input;
    cufftComplex* device_output;

    std::vector<float> output(2 * n * batch);

    cudaMalloc((void**)&device_input, sizeof(cufftComplex) * n * batch);
    cudaMalloc((void**)&device_output, sizeof(cufftComplex) * n * batch);

    cudaMemcpy(device_input, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice);

    cufftHandle fft_plan;

    cufftPlan1d(&fft_plan, n, CUFFT_C2C, batch);

    cufftExecC2C(fft_plan, device_input, device_output, CUFFT_FORWARD);
    cufftExecC2C(fft_plan, device_output, device_input, CUFFT_INVERSE);

    float norm_factor = 1.0f / n;

    cudaMemcpy(output.data(), device_input, sizeof(float) * output.size(), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] *= norm_factor;
    }

    cufftDestroy(fft_plan);
    cudaFree(device_input);
    cudaFree(device_output);

    return output;
}