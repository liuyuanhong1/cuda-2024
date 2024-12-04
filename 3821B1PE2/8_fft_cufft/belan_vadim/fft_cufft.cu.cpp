#include "fft_cufft.h"
#include <cufft.h>
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>

void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error occurred.");
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (input.size() % 2 != 0) {
        throw std::invalid_argument("Input size must be even (real and imaginary pairs).");
    }

    int n = input.size() / (2 * batch); 

    cufftComplex* d_input;
    cufftComplex* d_output;

    checkCudaError(cudaMalloc(&d_input, sizeof(cufftComplex) * n * batch));
    checkCudaError(cudaMalloc(&d_output, sizeof(cufftComplex) * n * batch));

    std::vector<cufftComplex> h_input(n * batch);
    for (int i = 0; i < n * batch; ++i) {
        h_input[i].x = input[2 * i];
        h_input[i].y = input[2 * i + 1];
    }

    checkCudaError(cudaMemcpy(d_input, h_input.data(), sizeof(cufftComplex) * n * batch, cudaMemcpyHostToDevice));

    cufftHandle plan;
    if (cufftPlan1d(&plan, n, CUFFT_C2C, batch) != CUFFT_SUCCESS) {
        throw std::runtime_error("CUFFT plan creation failed.");
    }

    if (cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        throw std::runtime_error("CUFFT forward transform failed.");
    }

    if (cufftExecC2C(plan, d_output, d_output, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        throw std::runtime_error("CUFFT inverse transform failed.");
    }

    cudaMemcpy(h_input.data(), d_output, sizeof(cufftComplex) * n * batch, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n * batch; ++i) {
        h_input[i].x /= n;
        h_input[i].y /= n;
    }

    std::vector<float> result(2 * n * batch);
    for (int i = 0; i < n * batch; ++i) {
        result[2 * i] = h_input[i].x;
        result[2 * i + 1] = h_input[i].y;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cufftDestroy(plan);

    return result;
}
