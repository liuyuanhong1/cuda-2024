#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

__global__ void normalize(cufftComplex* data, int num_elements, float norm_factor) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx].x *= norm_factor;
        data[idx].y *= norm_factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);

    if (input.size() != 2 * n * batch) {
        throw std::invalid_argument("Input size must be 2 * n * batch, where n is the number of elements per signal.");
    }

    cufftComplex* d_data;
    size_t input_size = input.size() * sizeof(float);

    cudaMalloc(&d_data, input_size);
    cudaMemcpy(d_data, input.data(), input_size, cudaMemcpyHostToDevice);

    cufftHandle plan;
    if (cufftPlan1d(&plan, n, CUFFT_C2C, batch) != CUFFT_SUCCESS) {
        cudaFree(d_data);
        throw std::runtime_error("CUFFT plan creation failed.");
    }

    if (cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("CUFFT execution for forward FFT failed.");
    }

    if (cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("CUFFT execution for inverse FFT failed.");
    }

    int num_elements = n * batch;
    float norm_factor = 1.0f / static_cast<float>(n);
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    normalize<<<blocksPerGrid, threadsPerBlock>>>(d_data, num_elements, norm_factor);

    std::vector<float> result(input.size());
    cudaMemcpy(result.data(), d_data, input_size, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_data);

    return result;
}
