// Copyright (c) 2024 Moiseev Nikita
#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

__global__ void normalizeKernel(float* output_data, int total_size, float normalization_factor) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < total_size) {
        output_data[index] *= normalization_factor;
    }
}

std::vector<float> FFTCUFFT(const std::vector<float>& input_data, int batch_size) {
    const int total_size = input_data.size();
    std::vector<float> normalized_output(total_size);
    int batch_elements = (total_size / batch_size) >> 1;

    int data_size_bytes = sizeof(cufftComplex) * batch_elements * batch_size;
    cufftHandle fft_plan;
    cufftPlan1d(&fft_plan, batch_elements, CUFFT_C2C, batch_size);
    cufftComplex* device_data;

    cudaMalloc(&device_data, data_size_bytes);
    cudaMemcpy(device_data, input_data.data(), data_size_bytes, cudaMemcpyHostToDevice);
    cufftExecC2C(fft_plan, device_data, device_data, CUFFT_FORWARD);
    cufftExecC2C(fft_plan, device_data, device_data, CUFFT_INVERSE);

    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);

    size_t threads_per_block = device_properties.maxThreadsPerBlock;
    size_t blocks_per_grid = (total_size + threads_per_block - 1) / threads_per_block;
    float normalization_factor = 1.0f / static_cast<float>(batch_elements);

    normalizeKernel<<<blocks_per_grid, threads_per_block>>>(
        reinterpret_cast<float*>(device_data), total_size, normalization_factor);

    cudaMemcpy(normalized_output.data(), device_data, data_size_bytes, cudaMemcpyDeviceToHost);
    cufftDestroy(fft_plan);
    cudaFree(device_data);

    return normalized_output;
}
