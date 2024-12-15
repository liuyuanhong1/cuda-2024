#include "fft_cufft.h"
#include <cufft.h>
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>

// Helper function to check CUDA errors
void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error occurred.");
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    // Check if the input size is valid
    if (input.size() % 2 != 0) {
        throw std::invalid_argument("Input size must be even (real and imaginary pairs).");
    }

    int n = input.size() / (2 * batch);  // n is the number of complex elements per signal

    // Step 1: Allocate memory for the input and output on the device
    cufftComplex* d_input;
    cufftComplex* d_output;

    checkCudaError(cudaMalloc(&d_input, sizeof(cufftComplex) * n * batch));
    checkCudaError(cudaMalloc(&d_output, sizeof(cufftComplex) * n * batch));

    // Step 2: Copy input data from host to device
    std::vector<cufftComplex> h_input(n * batch);
    for (int i = 0; i < n * batch; ++i) {
        h_input[i].x = input[2 * i];      // real part
        h_input[i].y = input[2 * i + 1];  // imaginary part
    }

    checkCudaError(cudaMemcpy(d_input, h_input.data(), sizeof(cufftComplex) * n * batch, cudaMemcpyHostToDevice));

    // Step 3: Create cuFFT plan for 1D complex-to-complex transform
    cufftHandle plan;
    if (cufftPlan1d(&plan, n, CUFFT_C2C, batch) != CUFFT_SUCCESS) {
        throw std::runtime_error("CUFFT plan creation failed.");
    }

    // Step 4: Perform the forward FFT (C2C transform)
    if (cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        throw std::runtime_error("CUFFT forward transform failed.");
    }

    // Step 5: Perform the inverse FFT (C2C transform)
    if (cufftExecC2C(plan, d_output, d_output, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        throw std::runtime_error("CUFFT inverse transform failed.");
    }

    // Step 6: Normalize the result by dividing by n
    cudaMemcpy(h_input.data(), d_output, sizeof(cufftComplex) * n * batch, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n * batch; ++i) {
        h_input[i].x /= n;
        h_input[i].y /= n;
    }

    // Step 7: Copy the result back to a flat vector of floats (real, imaginary pairs)
    std::vector<float> result(2 * n * batch);
    for (int i = 0; i < n * batch; ++i) {
        result[2 * i] = h_input[i].x;
        result[2 * i + 1] = h_input[i].y;
    }

    // Step 8: Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cufftDestroy(plan);

    return result;
}